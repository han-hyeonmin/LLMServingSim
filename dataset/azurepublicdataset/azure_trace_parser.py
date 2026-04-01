"""
azure_trace_parser.py

Converts Azure LLM Inference Public Trace CSV files into LLMServingSim
workload trace format (.jsonl). Token ID generation follows the same
approach used in sharegpt_parser.py.

Directory layout expected:
    dataset/
        azurepublicdataset/
            azure_trace_parser.py       <- this script
            AzureLLMInferenceTrace_conv.csv
            AzureLLMInferenceTrace_code.csv

Usage:
    # Convert all requests
    python dataset/azurepublicdataset/azure_trace_parser.py conv
    python dataset/azurepublicdataset/azure_trace_parser.py code

    # Limit to N requests
    python dataset/azurepublicdataset/azure_trace_parser.py conv --max-requests 100
    python dataset/azurepublicdataset/azure_trace_parser.py code --max-requests 500

Output filename examples:
    dataset/azure_trace_conv_llama.jsonl              (all requests)
    dataset/azure_trace_conv_req100_llama.jsonl       (100 requests)

Output record format (one JSON object per line):
    {
        "input_toks":      int,
        "output_toks":     int,
        "arrival_time_ns": int,
        "input_tok_ids":   List[int],
        "output_tok_ids":  List[int]
    }
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

# ==========================================
# Reproducibility  (mirrors sharegpt_parser.py)
# ==========================================
random.seed(42)
np.random.seed(42)

# ==========================================
# HF token  (mirrors sharegpt_parser.py)
# ==========================================
HF_TOKEN = getpass.getpass("Enter your Hugging Face token: ")
os.environ["HF_TOKEN"] = HF_TOKEN

# ==========================================
# Tokenizer
# Vocab size is read directly from the tokenizer instead of being
# hard-coded, so token IDs stay within the correct range.
# Llama-3.1-8B: 128256  (NOT 32000)
# ==========================================
TOKENIZER_NAME = "meta-llama/Llama-3.1-8B"
TOKENIZER_SUFFIX = "llama"  # appended to the output filename for identification

print(f"Loading tokenizer: {TOKENIZER_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, use_fast=True)
VOCAB_SIZE = tokenizer.vocab_size if hasattr(tokenizer, "vocab_size") else 32000
print(f"vocab_size = {VOCAB_SIZE}")

# ==========================================
# Column names defined in the Azure Public Dataset
# ==========================================
TIMESTAMP_COL = "TIMESTAMP"
INPUT_TOKENS_COL = "ContextTokens"
OUTPUT_TOKENS_COL = "GeneratedTokens"

# ==========================================
# Path configuration
# This script lives at dataset/azurepublicdataset/azure_trace_parser.py.
# CSV files sit in the same directory; output .jsonl files go one level up
# (dataset/).
# ==========================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # dataset/azurepublicdataset/
DATASET_DIR = os.path.dirname(SCRIPT_DIR)  # dataset/

CSV_FILES = {
    "conv": os.path.join(SCRIPT_DIR, "AzureLLMInferenceTrace_conv.csv"),
    "code": os.path.join(SCRIPT_DIR, "AzureLLMInferenceTrace_code.csv"),
}

# Requests whose input + output token count exceeds this limit are skipped
# to prevent simulator OOM errors
max_total_tokens = 4096


def build_output_path(key: str, max_requests) -> str:
    """
    Build the output .jsonl file path based on dataset key, request cap,
    and tokenizer suffix.

    Examples:
        conv, None  -> dataset/azure_trace_conv_llama.jsonl
        conv, 100   -> dataset/azure_trace_conv_req100_llama.jsonl
        code, 500   -> dataset/azure_trace_code_req500_llama.jsonl
    """
    req_tag = f"_req{max_requests}" if max_requests is not None else ""
    filename = f"azure_trace_{key}{req_tag}_{TOKENIZER_SUFFIX}.jsonl"
    return os.path.join(DATASET_DIR, filename)


# ==========================================
# Core conversion
# ==========================================
def convert_csv_to_jsonl(csv_path: str, output_path: str, max_requests) -> None:
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)

    # Sort requests in chronological order
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)

    # Compute arrival time in nanoseconds relative to the first request
    first_time = df[TIMESTAMP_COL].iloc[0]
    df["arrival_time_ns"] = (
        (df[TIMESTAMP_COL] - first_time).dt.total_seconds() * 1_000_000_000
    ).astype(int)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    request_count = 0
    skipped_count = 0

    print(f"Writing JSONL: {output_path}")
    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):

            if max_requests is not None and request_count >= max_requests:
                break

            input_len = int(row[INPUT_TOKENS_COL])
            output_len = int(row[OUTPUT_TOKENS_COL])

            # Skip rows with non-positive token counts
            if input_len <= 0 or output_len <= 0:
                skipped_count += 1
                continue

            # Skip requests that exceed the combined token limit
            if input_len + output_len > max_total_tokens:
                skipped_count += 1
                continue

            # Generate dummy token IDs using the same method as
            # sharegpt_parser.py (fix_len branch):
            #   random.randint(0, vocab_size - 1)  — uniform sampling
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

    print(f"Done. {request_count} requests written, {skipped_count} skipped.")


# ==========================================
# Entry point
# ==========================================
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
        help="Maximum number of requests to write (default: all)",
    )
    args = parser.parse_args()

    csv_path = CSV_FILES[args.dataset]
    output_path = build_output_path(args.dataset, args.max_requests)

    print(f"Output path: {output_path}")
    convert_csv_to_jsonl(csv_path, output_path, args.max_requests)


if __name__ == "__main__":
    main()
