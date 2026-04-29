"""Azure LLM Inference Trace CSV → LLMServingSim JSONL.

Canonical parser for the public Azure LLM inference traces (conv / code).
Follows the same ``register_args`` / ``run`` interface as ``sharegpt.py``
so it can be wired into ``workloads.generators.__main__`` identically::

    python -m workloads.generators azure --dataset conv --model meta-llama/Llama-3.1-8B \\
        --output workloads/azure_trace_conv_llama.jsonl

Output format (flat-request, same as ShareGPT):

    {
      "input_toks":      <int>,
      "output_toks":     <int>,
      "arrival_time_ns": <int>,
      "input_tok_ids":   [...],   # dummy uniform-random token IDs
      "output_tok_ids":  [...]    # dummy uniform-random token IDs
    }

The Azure CSVs contain real token counts but no token IDs, so
``input_tok_ids`` / ``output_tok_ids`` are filled with uniform-random
IDs (same strategy as the original parser) — sufficient for prefix-cache
hash testing when real IDs are not required.

Usage examples
--------------
    python -m workloads.generators azure \\
        --dataset conv --model meta-llama/Llama-3.1-8B \\
        --output workloads/azure_trace_conv_llama.jsonl

    python -m workloads.generators azure \\
        --dataset code --model meta-llama/Llama-3.1-8B \\
        --max-requests 100 \\
        --output workloads/azure_trace_code_req100_llama.jsonl

    python -m workloads.generators azure \\
        --dataset conv --model meta-llama/Llama-3.1-8B \\
        --all-arrives-at-0 \\
        --output workloads/azure_trace_conv_all_arrives_at_0_llama.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Azure CSV schema
# ---------------------------------------------------------------------------

TIMESTAMP_COL = "TIMESTAMP"
INPUT_TOKENS_COL = "ContextTokens"
OUTPUT_TOKENS_COL = "GeneratedTokens"

# Paths — CSVs are expected under workloads/azurepublicdataset/ by default,
# but can be overridden with --csv-dir.
_DEFAULT_CSV_DIR = Path(__file__).parent.parent / "azurepublicdataset"

CSV_NAMES = {
    "conv": "AzureLLMInferenceTrace_conv.csv",
    "code": "AzureLLMInferenceTrace_code.csv",
}


# ---------------------------------------------------------------------------
# CLI — register_args / run interface (mirrors sharegpt.py)
# ---------------------------------------------------------------------------


def register_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--dataset",
        choices=["conv", "code"],
        required=True,
        help="Which Azure trace to convert: 'conv' or 'code'.",
    )
    p.add_argument(
        "--model",
        required=True,
        help="HF model id or local path used as the tokenizer "
        "(vocab size is read from it; no GPU required).",
    )
    p.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    p.add_argument(
        "--csv-dir",
        dest="csv_dir",
        default=None,
        metavar="DIR",
        help=f"Directory containing the Azure CSV files. "
        f"Default: {_DEFAULT_CSV_DIR}",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for dummy token-ID generation. Default: 42.",
    )

    # ---- Request cap -------------------------------------------------------
    p.add_argument(
        "--max-requests",
        type=int,
        default=None,
        dest="max_requests",
        metavar="N",
        help="Cap on number of output requests (default: all).",
    )

    # ---- Length filters (names aligned with sharegpt.py) -------------------
    p.add_argument(
        "--max-input-toks",
        type=int,
        default=2048,
        dest="max_input_toks",
        metavar="N",
        help="Per-request input-token cap; should match "
        "--max-num-batched-tokens in the simulator (default: 2048).",
    )
    p.add_argument(
        "--max-kv-toks",
        type=int,
        default=4096,
        dest="max_kv_toks",
        metavar="N",
        help="Per-request input+output token cap, i.e. KV-memory upper bound "
        "(default: 4096).",
    )

    # ---- Arrival-time mode -------------------------------------------------
    p.add_argument(
        "--all-arrives-at-0",
        action="store_true",
        default=False,
        dest="all_arrives_at_0",
        help="Set every arrival_time_ns to 0 (batch / offline mode). "
        "Without this flag, real relative timestamps derived from the "
        "CSV TIMESTAMP column are used.",
    )


def run(args: argparse.Namespace) -> int:
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Resolve CSV path
    csv_dir = Path(args.csv_dir) if args.csv_dir else _DEFAULT_CSV_DIR
    csv_path = csv_dir / CSV_NAMES[args.dataset]
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Azure CSV not found: {csv_path}\n"
            f"Pass --csv-dir to point at the directory containing the file."
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Tokenizer — only vocab_size is needed; no GPU required
    tok = _load_tokenizer(args.model)
    vocab_size = getattr(tok, "vocab_size", 32000)
    print(f"Tokenizer: {args.model}  vocab_size={vocab_size}")

    # Load & sort CSV
    print(f"\nLoading CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    df[TIMESTAMP_COL] = pd.to_datetime(df[TIMESTAMP_COL])
    df = df.sort_values(by=TIMESTAMP_COL).reset_index(drop=True)

    # Compute arrival times
    if args.all_arrives_at_0:
        df["arrival_time_ns"] = 0
        print("  arrival_time_ns = 0  (--all-arrives-at-0)")
    else:
        first_ts = df[TIMESTAMP_COL].iloc[0]
        df["arrival_time_ns"] = (
            (df[TIMESTAMP_COL] - first_ts).dt.total_seconds() * 1_000_000_000
        ).astype(int)
        print("  arrival_time_ns = real relative timestamps")

    # Write JSONL
    print(f"\nWriting: {out_path}")
    print(f"  max_input_toks = {args.max_input_toks}")
    print(f"  max_kv_toks    = {args.max_kv_toks}")

    written = 0
    skipped = {"non_positive": 0, "input_too_large": 0, "kv_too_large": 0}

    with out_path.open("w", encoding="utf-8") as fout:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            if args.max_requests is not None and written >= args.max_requests:
                break

            input_len = int(row[INPUT_TOKENS_COL])
            output_len = int(row[OUTPUT_TOKENS_COL])

            if input_len <= 0 or output_len <= 0:
                skipped["non_positive"] += 1
                continue
            if input_len > args.max_input_toks:
                skipped["input_too_large"] += 1
                continue
            if input_len + output_len > args.max_kv_toks:
                skipped["kv_too_large"] += 1
                continue

            # Dummy uniform-random token IDs (mirrors sharegpt.py strategy)
            input_tok_ids = [
                random.randint(0, vocab_size - 1) for _ in range(input_len)
            ]
            output_tok_ids = [
                random.randint(0, vocab_size - 1) for _ in range(output_len)
            ]

            record = {
                "input_toks": input_len,
                "output_toks": output_len,
                "arrival_time_ns": int(row["arrival_time_ns"]),
                "input_tok_ids": input_tok_ids,
                "output_tok_ids": output_tok_ids,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    total_skipped = sum(skipped.values())
    print(f"\nDone.  {written} requests written,  {total_skipped} skipped.")
    print(f"  non-positive tokens  : {skipped['non_positive']}")
    print(f"  input > {args.max_input_toks:<5}        : {skipped['input_too_large']}")
    print(f"  input+output > {args.max_kv_toks:<5} : {skipped['kv_too_large']}")
    return 0


# ---------------------------------------------------------------------------
# Tokenizer helper (mirrors sharegpt.py)
# ---------------------------------------------------------------------------


def _load_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, use_fast=True, trust_remote_code=True)


# ---------------------------------------------------------------------------
# Standalone entry point (backward-compat; prefer python -m workloads.generators)
# ---------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(
        description="Convert Azure LLM Inference Trace CSV to LLMServingSim JSONL."
    )
    register_args(p)
    args = p.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
