"""
Convert LLMServingSim per-request CSV output to benchmark comparison format.
See README.md for column mapping and usage details.
"""

import argparse
import pandas as pd
from pathlib import Path

NS_TO_MS = 1e-6  # ns → ms
NS_TO_S = 1e-9  # ns → s

# All columns emitted by scheduler.py::save_output
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

# Subset required for conversion
REQUIRED_COLS = {"input", "output", "arrival", "end_time", "queuing_delay", "TTFT"}


def load_sim_csv(path: Path) -> pd.DataFrame:
    """Load and validate the LLMServingSim output CSV."""
    df = pd.read_csv(path)

    # Strip leading/trailing whitespace from column names (common with csv.writer)
    df.columns = df.columns.str.strip()

    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing from {path}: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )
    return df


def convert(df: pd.DataFrame) -> pd.DataFrame:
    """Transform LLMServingSim output rows into benchmark comparison format."""
    out = pd.DataFrame()

    out["request id"] = df["request id"].astype(int)
    out["L_prefill"] = df["input"].astype(int)
    out["L_decode"] = (df["output"] - df["input"]).astype(int)

    out["Decode start (s)"] = ((df["arrival"] + df["TTFT"]) * NS_TO_S).round(7)

    out["Decode end (s)"] = (df["end_time"] * NS_TO_S).round(7)

    out["Decode time (ms)"] = (
        (df["end_time"] - df["arrival"] - df["TTFT"]) * NS_TO_MS
    ).round(3)

    out["stall_total (ms)"] = (df["queuing_delay"] * NS_TO_MS).round(2)

    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert LLMServingSim per-request CSV output to benchmark comparison format.\n"
            "Computes decode token count (output - input), converts ns→ms/s, "
            "and calculates decode start/end times."
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

    neg_decode = (result_df["L_decode"] < 0).sum()
    if neg_decode:
        print(
            f"[WARN] {neg_decode} rows have L_decode < 0; "
            "check that 'output' column contains cumulative (input+decode) token count."
        )


if __name__ == "__main__":
    main()
