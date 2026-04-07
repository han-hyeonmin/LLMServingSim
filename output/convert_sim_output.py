"""
convert_sim_output.py

Converts LLMServingSim per-request CSV output into a format comparable
with the benchmark table shown in the reference screenshot.

Reference table columns (from screenshot):
  L_prefill   : number of input (prefill) tokens
  L_decode    : number of decode tokens  = output_col - input_col  (out - in)
  Decode start: time when decode phase begins, in seconds (= arrival + TTFT)
  Decode end  : time when decode phase ends,   in seconds (= arrival + latency)
  Decode time : total decode duration in ms    (= latency - TTFT, converted ns→ms)
  stall_total : total stall time in ms         (= queuing_delay, converted ns→ms)
  stall_ratio : stall_total / Decode time * 100  (%)

LLMServingSim output CSV columns (scheduler.py::save_output):
  instance id, request id, model, input, output,
  arrival, end_time, latency, queuing_delay, TTFT, TPOT, ITL

Time unit in LLMServingSim CSV: nanoseconds (ns)
"""

import argparse
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NS_TO_MS = 1e-6  # nanoseconds  → milliseconds
NS_TO_S = 1e-9  # nanoseconds  → seconds

# Columns emitted by scheduler.py::save_output
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

# Columns we actually need for the conversion
REQUIRED_COLS = {"input", "output", "arrival", "latency", "queuing_delay", "TTFT"}


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
    """
    Transform LLMServingSim output rows into the benchmark comparison format.

    All LLMServingSim time values are in nanoseconds (ns).

    Column derivations
    ------------------
    request id   = request id                        (carried through for sorting)
    L_prefill    = input                             (tokens)
    L_decode     = output - input                    (tokens)  [out-in trick]
    end_time     = end_time * NS_TO_S                (seconds)
    latency      = latency * NS_TO_S                 (seconds)
    stall_total  = queuing_delay      * NS_TO_MS     (ms)
    """
    out = pd.DataFrame()

    out["request id"] = df["request id"].astype(int)
    out["L_prefill"] = df["input"].astype(int)
    out["L_decode"] = (df["output"] - df["input"]).astype(int)

    # end_time in seconds
    out["end_time (s)"] = (df["end_time"] * NS_TO_S).round(7)

    # request latency in seconds
    out["latency (s)"] = (df["latency"] * NS_TO_S).round(7)

    # Decode duration and stall in milliseconds
    stall_total_ms = df["queuing_delay"] * NS_TO_MS

    out["stall_total (ms)"] = stall_total_ms.round(2)

    return out


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert LLMServingSim per-request CSV output to benchmark comparison format.\n"
            "Computes decode token count (output - input), converts ns→ms/s, "
            "and calculates stall ratio."
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

    # --- Load ---
    sim_df = load_sim_csv(args.input_csv)
    print(f"[INFO] Loaded {len(sim_df)} rows from {args.input_csv}")

    # --- Convert ---
    result_df = convert(sim_df)

    # Sort to match the time-ordered benchmark table
    if args.sort_by in result_df.columns:
        result_df = result_df.sort_values(args.sort_by).reset_index(drop=True)
    else:
        print(f"[WARN] Sort column '{args.sort_by}' not found; skipping sort.")

    # Remove req_id column
    result_df.drop(columns=["request id"], inplace=True)

    # --- Output path ---
    out_path = args.output or (
        args.input_csv.parent / (args.input_csv.stem + "_converted.csv")
    )

    result_df.to_csv(out_path, index=False)
    print(f"[INFO] Saved {len(result_df)} rows to {out_path}")

    # --- Preview ---
    print("\n[PREVIEW] First 5 rows:")
    print(result_df.head(5).to_string(index=False))

    # --- Quick sanity check ---
    neg_decode = (result_df["L_decode"] < 0).sum()
    if neg_decode:
        print(
            f"[WARN] {neg_decode} rows have L_decode < 0; "
            "check that 'output' column contains cumulative (input+decode) token count."
        )


if __name__ == "__main__":
    main()
