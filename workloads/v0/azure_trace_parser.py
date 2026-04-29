"""
azure_trace_parser.py — Azure LLM Inference Trace CSV → LLMServingSim JSONL
[LEGACY — superseded by workloads/generators/azure_trace_parser.py]

================================================================================
README (archived)
================================================================================

# Azure Trace Parser

Converts Azure LLM Inference Public Trace
(https://github.com/Azure/AzurePublicDataset) CSV files into the `.jsonl`
workload trace format consumed by LLMServingSim.

## Quick Start

    # Convert the conversation trace (all valid requests)
    python dataset/azurepublicdataset/azure_trace_parser.py conv

    # Convert the code trace, limited to 100 requests
    python dataset/azurepublicdataset/azure_trace_parser.py code --max-requests 100

You will be prompted for a Hugging Face token (required for the gated
Llama-3.1-8B tokenizer).

## Arguments

    dataset (positional)   conv or code
    --max-requests N       Cap the number of output requests          (default: all)
    --max-input-tokens N   Drop requests whose input_toks > N         (default: 2048)
    --max-total-tokens N   Drop requests whose input+output_toks > N  (default: 4096)
    --all-arrives-at-0     Set all arrival_time_ns to 0 (batch mode)

### How to choose --max-input-tokens

    max-input-tokens  ≤  main.py --max-num-batched-tokens   (scheduler upper bound)
    max-input-tokens  ≤  layer.sh --max-len                  (layers.csv coverage)

With the current project defaults (--max-num-batched-tokens 2048, --max-len 6000),
the effective ceiling is min(2048, 6000) = 2048.

## Output

Files are written to the dataset/ directory (one level above the script).

    … conv                    →  dataset/azure_trace_conv_llama.jsonl
    … conv --max-requests 100 →  dataset/azure_trace_conv_req100_llama.jsonl
    … code --max-requests 500 →  dataset/azure_trace_code_req500_llama.jsonl

Each output line:
    {
        "input_toks":      128,
        "output_toks":     64,
        "arrival_time_ns": 350000000,
        "input_tok_ids":   [91012, 4837, ...],
        "output_tok_ids":  [7231, 55902, ...]
    }

## Request Filtering

    raw CSV row
      ├── input_toks ≤ 0 or output_toks ≤ 0         →  SKIP  (data quality)
      ├── input_toks > --max-input-tokens            →  SKIP  (scheduler / profile coverage)
      ├── input_toks + output_toks > --max-total     →  SKIP  (simulator OOM prevention)
      └── PASS  →  written to .jsonl

## Directory Layout

    dataset/
    ├── azurepublicdataset/
    │   ├── azure_trace_parser.py            ← this script
    │   ├── README.md                        ← archived into this docstring
    │   ├── AzureLLMInferenceTrace_conv.csv
    │   └── AzureLLMInferenceTrace_code.csv
    ├── azure_trace_conv_llama.jsonl
    └── sharegpt_req100_rate10_llama.jsonl

## End-to-End Example

    # 1. Generate the trace
    python dataset/azurepublicdataset/azure_trace_parser.py conv \\
        --max-requests 100 \\
        --max-input-tokens 2048

    # 2. Run the simulator
    python main.py \\
        --cluster-config cluster_config/dojo_pd_3node.json \\
        --dataset dataset/azure_trace_conv_req100_llama.jsonl \\
        --max-num-batched-tokens 2048 \\
        --max-batch 5 \\
        --num-req 100 \\
        --log-level INFO

================================================================================
"""

import os
import json
import random
import getpass
import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer

# Reproducibility (mirrors sharegpt_parser.py)
random.seed(42)
np.random.seed(42)

# HF token (Llama-3.1-8B is a gated model)
HF_TOKEN = getpass.getpass("Enter your Hugging Face token: ")
os.environ["HF_TOKEN"] = HF_TOKEN

# Tokenizer — vocab size from tokenizer directly (Llama-3.1-8B: 128256)
TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"
TOKENIZER_SUFFIX = "llama"

print(f"Loading tokenizer: {TOKENIZER_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
VOCAB_SIZE = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 32000
print(f"vocab_size = {VOCAB_SIZE}")

# Azure CSV column names
TIMESTAMP_COL = "TIMESTAMP"
INPUT_TOKENS_COL = "ContextTokens"
OUTPUT_TOKENS_COL = "GeneratedTokens"

# Paths — script in workloads/v0/, CSVs in workloads/azurepublicdataset/, output goes to workloads/v0/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKLOADS_DIR = os.path.dirname(SCRIPT_DIR)  # workloads/
AZURE_CSV_DIR = os.path.join(WORKLOADS_DIR, "azurepublicdataset")
DATASET_DIR = SCRIPT_DIR  # workloads/v0/

CSV_FILES = {
    "conv": os.path.join(AZURE_CSV_DIR, "AzureLLMInferenceTrace_conv.csv"),
    "code": os.path.join(AZURE_CSV_DIR, "AzureLLMInferenceTrace_code.csv"),
}


def build_output_path(key: str, max_requests, all_arrives_at_0: bool) -> str:
    """Build output .jsonl path.
    e.g. conv, 100, True  → azure_trace_conv_req100_all_arrives_at_0_llama.jsonl
         conv, None, False → azure_trace_conv_llama.jsonl
    """
    req_tag = f"_req{max_requests}" if max_requests is not None else ""
    zero_tag = "_all_arrives_at_0" if all_arrives_at_0 else ""
    filename = f"azure_trace_{key}{req_tag}{zero_tag}_{TOKENIZER_SUFFIX}.jsonl"
    return os.path.join(DATASET_DIR, filename)


def convert_csv_to_jsonl(
    csv_path: str,
    output_path: str,
    max_requests,
    max_input_tokens: int,
    max_total_tokens: int,
    all_arrives_at_0: bool = False,
) -> None:
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Sort chronologically, compute relative arrival time (ns)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)
    first_time = df[TIMESTAMP_COL].iloc[0]
    if all_arrives_at_0:
        df["arrival_time_ns"] = 0
        print("  arrival_time_ns   = 0 (all_arrives_at_0 flag enabled)")
    else:
        df["arrival_time_ns"] = (
            (df[TIMESTAMP_COL] - first_time).dt.total_seconds() * 1_000_000_000
        ).astype(int)
        print("  arrival_time_ns   = real relative timestamps")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    request_count = 0
    skipped_count = 0
    skipped_input_too_large = 0
    skipped_total_too_large = 0
    skipped_non_positive = 0

    print(f"Writing JSONL: {output_path}")
    print(f"  max_input_tokens  = {max_input_tokens}")
    print(f"  max_total_tokens  = {max_total_tokens}")

    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):

            if max_requests is not None and request_count >= max_requests:
                break

            input_len = int(row[INPUT_TOKENS_COL])
            output_len = int(row[OUTPUT_TOKENS_COL])

            if input_len <= 0 or output_len <= 0:
                skipped_non_positive += 1
                skipped_count += 1
                continue

            if input_len > max_input_tokens:
                skipped_input_too_large += 1
                skipped_count += 1
                continue

            if input_len + output_len > max_total_tokens:
                skipped_total_too_large += 1
                skipped_count += 1
                continue

            # Dummy token IDs — uniform sampling, same as sharegpt_parser.py
            input_tok_ids = [
                random.randint(0, VOCAB_SIZE - 1) for _ in range(input_len)
            ]
            output_tok_ids = [
                random.randint(0, VOCAB_SIZE - 1) for _ in range(output_len)
            ]

            record = {
                "input_toks": input_len,
                "output_toks": output_len,
                "arrival_time_ns": int(row["arrival_time_ns"]),
                "input_tok_ids": input_tok_ids,
                "output_tok_ids": output_tok_ids,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            request_count += 1

    print(f"\nDone. {request_count} requests written, {skipped_count} skipped.")
    print(f"  Skipped breakdown:")
    print(f"    non-positive tokens   : {skipped_non_positive}")
    print(f"    input > {max_input_tokens:>5}         : {skipped_input_too_large}")
    print(f"    input+output > {max_total_tokens:>5}   : {skipped_total_too_large}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Azure LLM Inference Trace CSV to LLMServingSim JSONL."
    )
    parser.add_argument(
        "dataset",
        choices=["conv", "code"],
        help="Which trace to convert: 'conv' or 'code'",
    )
    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        metavar="N",
        help="Cap on number of output requests (default: all)",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=2048,
        metavar="N",
        help="Per-request input token cap; match --max-num-batched-tokens in main.py (default: 2048)",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=4096,
        metavar="N",
        help="Per-request input+output token cap (default: 4096)",
    )
    parser.add_argument(
        "--all-arrives-at-0",
        action="store_true",
        default=False,
        help="Set all arrival_time_ns to 0 (batch mode). "
        "Without this flag, real relative timestamps are used.",
    )
    args = parser.parse_args()

    csv_path = CSV_FILES[args.dataset]
    output_path = build_output_path(
        args.dataset, args.max_requests, args.all_arrives_at_0
    )

    print(f"Output path: {output_path}")
    convert_csv_to_jsonl(
        csv_path,
        output_path,
        args.max_requests,
        args.max_input_tokens,
        args.max_total_tokens,
        all_arrives_at_0=args.all_arrives_at_0,
    )


if __name__ == "__main__":
    main()
