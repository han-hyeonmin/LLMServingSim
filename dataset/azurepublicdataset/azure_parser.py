import pandas as pd
import json
import random
from tqdm import tqdm
import os
import sys

# ==========================================
# Configuration
# ==========================================
# Define CSV/JSONL file mapping
file_pairs = {
    "conv": {"csv": "AzureLLMInferenceTrace_conv.csv", "jsonl": "../azure_trace_conv.jsonl"},
    "code": {"csv": "AzureLLMInferenceTrace_code.csv", "jsonl": "../azure_trace_code.jsonl"}
}

# Column names defined in the Azure Public Dataset
TIMESTAMP_COL = "TIMESTAMP"
INPUT_TOKENS_COL = "ContextTokens"
OUTPUT_TOKENS_COL = "GeneratedTokens"

vocab_size = 32000        # Dummy vocabulary size for token ID generation
max_requests = 1000       # Maximum number of valid requests to extract
max_total_tokens = 4096   # Skip requests exceeding this length to prevent simulator OOM

# ==========================================
# Reproducibility
# ==========================================
random.seed(42)

def convert_csv_to_jsonl(csv_file_path, output_path):
    """
    Convert a single CSV trace file to JSONL format.
    """
    print(f"Loading CSV file from {csv_file_path}...")
    df = pd.read_csv(csv_file_path)

    # Parse datetime strings and sort requests chronologically
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(by=TIMESTAMP_COL)

    # Calculate the relative arrival time offset (in seconds) from the very first request
    first_time = df[TIMESTAMP_COL].iloc[0]
    df['time_offset_s'] = (df[TIMESTAMP_COL] - first_time).dt.total_seconds()

    request_count = 0

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # ==========================================
    # Generate JSONL Trace
    # ==========================================
    print(f"Generating JSONL trace to {output_path}...")
    with open(output_path, "w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            if request_count >= max_requests:
                break

            input_len = int(row[INPUT_TOKENS_COL])
            output_len = int(row[OUTPUT_TOKENS_COL])

            # Filter out invalid requests (length <= 0)
            if input_len <= 0 or output_len <= 0:
                continue

            # Filter out excessively long requests that might break the simulator's memory bounds
            if input_len + output_len > max_total_tokens:
                continue

            # Convert the time offset from seconds to nanoseconds (ns)
            arrival_time_ns = int(row['time_offset_s'] * 1_000_000_000)

            # Generate dummy token IDs (Azure trace lacks actual text/tokens)
            input_tokens = [random.randint(0, vocab_size - 1) for _ in range(input_len)]
            output_tokens = [random.randint(0, vocab_size - 1) for _ in range(output_len)]

            # Construct the request record conforming to LLMServingSim's standard
            record = {
                "input_toks": input_len,
                "output_toks": output_len,
                "arrival_time_ns": arrival_time_ns,
                "input_tok_ids": input_tokens,
                "output_tok_ids": output_tokens
            }

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            request_count += 1

    print(f"Successfully converted {request_count} requests.\n")

def main():
    # ==========================================
    # Parse command-line argument
    # Usage: python script.py conv
    #        python script.py code
    # ==========================================
    if len(sys.argv) != 2 or sys.argv[1] not in file_pairs:
        print(f"Usage: python {sys.argv[0]} conv|code")
        sys.exit(1)

    key = sys.argv[1]
    convert_csv_to_jsonl(file_pairs[key]["csv"], file_pairs[key]["jsonl"])

if __name__ == "__main__":
    main()