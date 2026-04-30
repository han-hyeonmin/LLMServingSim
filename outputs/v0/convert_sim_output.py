"""
convert_sim_output.py
=====================
Convert LLMServingSim per-request CSV output to benchmark comparison format.

This script is the sole documentation source; README.md is no longer needed
and can be deleted after verifying this file.

────────────────────────────────────────────────────────────────────────────
Directory context
────────────────────────────────────────────────────────────────────────────
  outputs/
    *.csv                   # Per-request results written by main.py --output
    v0/
      convert_sim_output.py # This script (current location)
      *.csv                 # Legacy simulation run results

────────────────────────────────────────────────────────────────────────────
Column mapping  (all raw time values from scheduler.py::save_output are in ns)
────────────────────────────────────────────────────────────────────────────
  Output column        Unit   Derivation
  ─────────────────── ──────  ──────────────────────────────────────────────
  L_prefill            tokens  input
  L_decode             tokens  output - input
  Decode start (s)     s       (arrival + TTFT) × 1e-9
  Decode end (s)       s       end_time × 1e-9
  Decode time (ms)     ms      (end_time - arrival - TTFT) × 1e-6
  stall_total (ms)     ms      queuing_delay × 1e-6

────────────────────────────────────────────────────────────────────────────
Usage
────────────────────────────────────────────────────────────────────────────
  # Default: writes <stem>_converted.csv in the same directory
  python outputs/v0/convert_sim_output.py outputs/example_run.csv

  # Specify output path
  python outputs/v0/convert_sim_output.py outputs/example_run.csv -o outputs/result_bench.csv

  # Change sort column (default: request id)
  python outputs/v0/convert_sim_output.py outputs/example_run.csv --sort-by "Decode start (s)"

────────────────────────────────────────────────────────────────────────────
Notes
────────────────────────────────────────────────────────────────────────────
  - The `output` column must be a cumulative token count (input + decode tokens).
    Rows with L_decode < 0 are flagged with a warning but not dropped.
  - The script exits with an error if any required column is missing:
    input, output, arrival, end_time, queuing_delay, TTFT
"""

import argparse
import pandas as pd
from pathlib import Path

# ── Unit conversion factors ──────────────────────────────────────────────────
NS_TO_MS = 1e-6  # nanoseconds → milliseconds
NS_TO_S = 1e-9  # nanoseconds → seconds

# ── Full column set emitted by scheduler.py::save_output ────────────────────
SIM_COLS = [
    "instance id",
    "request id",
    "model",
    "input",
    "output",
    "arrival",
    "end_time",
    "latency",
    "queuing_delay",
    "TTFT",
    "TPOT",
    "ITL",
]

# ── Subset actually required for the conversion below ────────────────────────
REQUIRED_COLS = {"input", "output", "arrival", "end_time", "queuing_delay", "TTFT"}


def load_sim_csv(path: Path) -> pd.DataFrame:
    """Load and validate the LLMServingSim output CSV.

    Strips whitespace from column names (common artifact of csv.writer) and
    raises ValueError when any required column is absent.
    """
    df = pd.read_csv(path)

    # csv.writer sometimes pads column names with spaces — normalise them
    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing from {path}: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    return df


def convert(df: pd.DataFrame) -> pd.DataFrame:
    """Transform LLMServingSim output rows into benchmark comparison format.

    All source time values are assumed to be in nanoseconds (ns), as written
    by scheduler.py::save_output.  Decode start is defined as the moment the
    first decode step begins, i.e. arrival + TTFT.
    """
    out = pd.DataFrame()

    out["request id"] = df["request id"].astype(int)
    out["L_prefill"] = df["input"].astype(int)

    # output column is cumulative (input + decode tokens), so subtract input
    out["L_decode"] = (df["output"] - df["input"]).astype(int)

    # Decode phase start: when TTFT has elapsed after the request arrived
    out["Decode start (s)"] = ((df["arrival"] + df["TTFT"]) * NS_TO_S).round(7)

    out["Decode end (s)"] = (df["end_time"] * NS_TO_S).round(7)

    # Decode duration excludes the prefill (TTFT) portion
    out["Decode time (ms)"] = (
        (df["end_time"] - df["arrival"] - df["TTFT"]) * NS_TO_MS
    ).round(3)

    # Total time the request spent waiting in the queue
    out["stall_total (ms)"] = (df["queuing_delay"] * NS_TO_MS).round(2)

    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert LLMServingSim per-request CSV output to benchmark comparison format.\n"
            "Computes decode token count (output - input), converts ns→ms/s, "
            "and calculates decode start/end times.\n\n"
            "README.md has been merged into this file's docstring; it is safe to delete."
        )
    )
    parser.add_argument(
        "input_csv",
        type=Path,
        help="Path to LLMServingSim output CSV (e.g. output/example_run.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Path for converted output CSV. "
            "Defaults to <input_stem>_converted.csv in the same directory."
        ),
    )
    parser.add_argument(
        "--sort-by",
        default="request id",
        help="Column to sort the output by (default: 'request id')",
    )
    args = parser.parse_args()

    sim_df = load_sim_csv(args.input_csv)
    print(f"[INFO] Loaded {len(sim_df)} rows from {args.input_csv}")

    result_df = convert(sim_df)

    if args.sort_by in result_df.columns:
        result_df = result_df.sort_values(args.sort_by).reset_index(drop=True)
    else:
        print(f"[WARN] Sort column '{args.sort_by}' not found; skipping sort.")

    out_path = args.output or (
        args.input_csv.parent / (args.input_csv.stem + "_converted.csv")
    )

    result_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {len(result_df)} rows to {out_path}")

    print("\n[PREVIEW] First 5 rows:")
    print(result_df.head(5).to_string(index=False))

    # Sanity-check: negative L_decode means 'output' was not cumulative
    neg_decode = (result_df["L_decode"] < 0).sum()
    if neg_decode:
        print(
            f"[WARN] {neg_decode} rows have L_decode < 0; "
            "check that 'output' column contains cumulative (input+decode) token count."
        )


if __name__ == "__main__":
    main()
