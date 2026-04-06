"""
azure_trace_parser.py — Azure LLM Inference Trace CSV → LLMServingSim JSONL

Usage:
    python dataset/azurepublicdataset/azure_trace_parser.py conv
    python dataset/azurepublicdataset/azure_trace_parser.py code --max-requests 100
    python dataset/azurepublicdataset/azure_trace_parser.py conv --max-input-tokens 4096

Output: dataset/azure_trace_{conv|code}[_reqN]_llama.jsonl
"""

import os
import sys
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

# Paths — script in dataset/azurepublicdataset/, output goes to dataset/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.dirname(SCRIPT_DIR)

CSV_FILES = {
    "conv": os.path.join(SCRIPT_DIR, "AzureLLMInferenceTrace_conv.csv"),
    "code": os.path.join(SCRIPT_DIR, "AzureLLMInferenceTrace_code.csv"),
}


def build_output_path(key: str, max_requests) -> str:
    """Build output .jsonl path. e.g. conv, 100 → azure_trace_conv_req100_llama.jsonl"""
    req_tag = f"_req{max_requests}" if max_requests is not None else ""
    filename = f"azure_trace_{key}{req_tag}_{TOKENIZER_SUFFIX}.jsonl"
    return os.path.join(DATASET_DIR, filename)


def convert_csv_to_jsonl(
    csv_path: str,
    output_path: str,
    max_requests,
    max_input_tokens: int,
    max_total_tokens: int,
) -> None:
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Sort chronologically, compute relative arrival time (ns)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)
    first_time = df[TIMESTAMP_COL].iloc[0]
    df["arrival_time_ns"] = (
        (df[TIMESTAMP_COL] - first_time).dt.total_seconds() * 1_000_000_000
    ).astype(int)

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

            # Drop if input alone exceeds scheduler's max_num_batched_tokens,
            # which also prevents _get_perf_row KeyError on layers.csv miss.
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
    args = parser.parse_args()

    csv_path = CSV_FILES[args.dataset]
    output_path = build_output_path(args.dataset, args.max_requests)

    print(f"Output path: {output_path}")
    convert_csv_to_jsonl(
        csv_path,
        output_path,
        args.max_requests,
        args.max_input_tokens,
        args.max_total_tokens,
    )


if __name__ == "__main__":
    main()
