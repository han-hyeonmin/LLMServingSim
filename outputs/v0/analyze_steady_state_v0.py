"""
Steady-state analysis for LLMServingSim v0 per-request CSV output.

Reconstructs per-token decode completion timestamps from the ITL list, builds
a concurrent-decode timeseries, detects the longest steady-state window, and
reports throughput / TPOT metrics confined to that window.

ITL semantics (verified against v0 data, std=0 across 100 requests):
    - Unit: nanoseconds
    - len(ITL) == L_decode - 1
    - sum(ITL) == latency - TTFT  (exact)
    - token[0] completes at: arrival + TTFT
    - token[k] completes at: arrival + TTFT + sum(ITL[0..k-1])  for k >= 1

Usage:
    python analyze_steady_state_v0.py <input_csv> [--grid-ms 100] [--theta 0.9]
                                                  [--hysteresis-s 2.0] [--plot]
                                                  [--out-dir <dir>] [--verbose]
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000

# v0 CSV columns we expect; see scheduler.py::save_output
REQUIRED_COLS = [
    "instance id",
    "request id",
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


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

logger = logging.getLogger("steady_state")


def configure_logger(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


# ---------------------------------------------------------------------------
# CSV loading and validation
# ---------------------------------------------------------------------------


def load_v0_csv(path: Path) -> pd.DataFrame:
    """Load v0 CSV and assert the required schema."""
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    # Cast numeric columns up front so downstream comparisons don't surprise us
    int_cols = [
        "instance id",
        "request id",
        "input",
        "output",
        "arrival",
        "end_time",
        "latency",
        "queuing_delay",
        "TTFT",
        "TPOT",
    ]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")

    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def parse_itl_lists(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the ITL string column into a list[int] column (in place).

    Handles:
      - Empty list "[]" or empty string  -> []
      - Standard repr "[123, 456, ...]"  -> [123, 456, ...]
    """

    def _parse(s):
        # ITL can be missing or empty for prefill-only rows or 1-token decodes
        if pd.isna(s):
            return []
        s = str(s).strip()
        if s == "" or s == "[]":
            return []
        try:
            v = ast.literal_eval(s)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse ITL string: {s!r} ({e})") from e
        if not isinstance(v, list):
            raise ValueError(f"ITL is not a list: {s!r}")
        return [int(x) for x in v]

    df = df.copy()
    df["ITL_list"] = df["ITL"].apply(_parse)
    df["ITL_len"] = df["ITL_list"].apply(len)
    df["L_decode"] = df["output"] - df["input"]
    return df


# ---------------------------------------------------------------------------
# Self-check / safety guards
# ---------------------------------------------------------------------------


@dataclass
class SelfCheckReport:
    """Summary of the data-integrity checks before analysis."""

    n_rows: int
    n_empty_itl: int
    n_zero_first_itl: int  # candidate full-prefix-cache-hit rows
    n_len_mismatch: int  # ITL_len != L_decode - 1
    n_sum_mismatch: int  # sum(ITL) != latency - TTFT
    instance_ids: list[int]
    rows_per_instance: dict
    decode_window_s: tuple[float, float]


def run_self_check(df: pd.DataFrame, *, strict: bool) -> SelfCheckReport:
    """Verify ITL invariants and report edge cases.

    The three identities below were verified to hold with std=0 on the
    reference v0 dataset. We re-check them here so the analysis fails loudly
    if the input CSV violates them.

    strict=True: raise on any violation
    strict=False: log a warning and continue (used when the user knows the CSV
                  contains prefill-only rows, e.g. PD-disagg)
    """
    n_rows = len(df)

    # Empty-ITL rows can appear for: prefill-only rows in PD-disagg, or
    # 1-token-decode requests (L_decode == 1, so ITL has length 0)
    empty_mask = df["ITL_len"] == 0
    n_empty_itl = int(empty_mask.sum())

    # First ITL == 0 indicates the full-prefix-cache-hit path:
    # set_ttft and add_itl fire in the same step, so ITL[0] == 0.
    def _first_zero(lst):
        return len(lst) > 0 and lst[0] == 0

    n_zero_first_itl = int(df["ITL_list"].apply(_first_zero).sum())

    # Identity 1: len(ITL) == L_decode - 1
    expected_len = (df["L_decode"] - 1).clip(lower=0)
    len_mismatch = df["ITL_len"] != expected_len
    n_len_mismatch = int(len_mismatch.sum())

    # Identity 2: sum(ITL) == latency - TTFT
    df_local = df.copy()
    df_local["ITL_sum"] = df_local["ITL_list"].apply(sum)
    sum_diff = (df_local["latency"] - df_local["TTFT"]) - df_local["ITL_sum"]
    sum_mismatch = sum_diff != 0
    n_sum_mismatch = int(sum_mismatch.sum())

    # Per-instance breakdown helps spot PD-disaggregation
    inst_ids = sorted(df["instance id"].unique().tolist())
    rows_per_inst = df.groupby("instance id").size().to_dict()
    rows_per_inst = {int(k): int(v) for k, v in rows_per_inst.items()}

    # Overall decode time window (used downstream as a sanity bound)
    decode_start_min = int((df["arrival"] + df["TTFT"]).min())
    decode_end_max = int(df["end_time"].max())
    win = (decode_start_min / NS_PER_S, decode_end_max / NS_PER_S)

    report = SelfCheckReport(
        n_rows=n_rows,
        n_empty_itl=n_empty_itl,
        n_zero_first_itl=n_zero_first_itl,
        n_len_mismatch=n_len_mismatch,
        n_sum_mismatch=n_sum_mismatch,
        instance_ids=[int(x) for x in inst_ids],
        rows_per_instance=rows_per_inst,
        decode_window_s=win,
    )

    logger.info("Self-check: %d rows total", report.n_rows)
    logger.info("  instance ids: %s", report.instance_ids)
    logger.info("  rows/instance: %s", report.rows_per_instance)
    logger.info("  empty ITL rows: %d", report.n_empty_itl)
    logger.info("  ITL[0]==0 rows (full prefix hit?): %d", report.n_zero_first_itl)
    logger.info("  len(ITL) mismatch: %d", report.n_len_mismatch)
    logger.info("  sum(ITL) mismatch: %d", report.n_sum_mismatch)
    logger.info(
        "  decode time span (s): %.3f -> %.3f",
        report.decode_window_s[0],
        report.decode_window_s[1],
    )

    # Surface the offending rows in DEBUG so the user can investigate
    if n_len_mismatch > 0:
        bad = df_local.loc[
            len_mismatch,
            ["instance id", "request id", "input", "output", "L_decode", "ITL_len"],
        ].head(5)
        logger.debug("Sample len mismatches:\n%s", bad.to_string())
    if n_sum_mismatch > 0:
        bad = df_local.loc[
            sum_mismatch, ["instance id", "request id", "latency", "TTFT", "ITL_sum"]
        ].head(5)
        logger.debug("Sample sum mismatches:\n%s", bad.to_string())

    if strict and (n_len_mismatch > 0 or n_sum_mismatch > 0):
        raise ValueError(
            f"Self-check failed under strict mode: "
            f"len_mismatch={n_len_mismatch}, sum_mismatch={n_sum_mismatch}. "
            f"Run with --no-strict to continue with warnings."
        )

    return report


# ---------------------------------------------------------------------------
# Per-token event reconstruction
# ---------------------------------------------------------------------------


def expand_decode_events(df: pd.DataFrame) -> pd.DataFrame:
    """Expand each request into per-decode-token completion events.

    Returns a DataFrame with columns:
        instance_id, request_id, token_idx, t_ns

    For a request with L_decode tokens, emits L_decode events:
        token_idx=0 at  arrival + TTFT
        token_idx=k at  arrival + TTFT + sum(ITL[0..k-1])  for k >= 1

    Rows with L_decode <= 0 are skipped (prefill-only rows produce no events).
    """
    rows = []
    skipped_no_decode = 0

    for _, r in df.iterrows():
        l_decode = int(r["L_decode"])
        if l_decode <= 0:
            skipped_no_decode += 1
            continue

        arrival = int(r["arrival"])
        ttft = int(r["TTFT"])
        itl = r["ITL_list"]
        inst_id = int(r["instance id"])
        req_id = int(r["request id"])

        # Defensive: ITL length should match L_decode - 1; if not, we still
        # emit what we can rather than crash. The self-check already flagged
        # this case; here we degrade gracefully for plotting.
        expected_itl_len = l_decode - 1
        if len(itl) != expected_itl_len:
            logger.warning(
                "request_id=%d: ITL length %d != expected %d; truncating/padding",
                req_id,
                len(itl),
                expected_itl_len,
            )
            itl = itl[:expected_itl_len]  # truncate over-long
            # If under-length, the loop below just emits fewer events.

        t = arrival + ttft
        rows.append((inst_id, req_id, 0, t))
        for k, dt in enumerate(itl):
            t += int(dt)
            rows.append((inst_id, req_id, k + 1, t))

    if skipped_no_decode > 0:
        logger.info(
            "Skipped %d rows with L_decode<=0 (prefill-only?)", skipped_no_decode
        )

    events = pd.DataFrame(
        rows, columns=["instance_id", "request_id", "token_idx", "t_ns"]
    )
    events = events.sort_values("t_ns").reset_index(drop=True)
    logger.info("Expanded %d decode token events", len(events))
    return events


# ---------------------------------------------------------------------------
# Concurrent-decode timeseries
# ---------------------------------------------------------------------------


def build_concurrent_decode_ts(
    df: pd.DataFrame,
    grid_ms: float,
    *,
    instance_filter: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Build a uniform-grid timeseries of concurrent decodes.

    Each request occupies the half-open interval [decode_start, decode_end),
    where decode_start = arrival + TTFT and decode_end = end_time. We sample
    on a uniform grid and count overlapping intervals using a sweep-line.

    Args:
        df: Parsed v0 dataframe (must include arrival, TTFT, end_time, L_decode).
        grid_ms: Grid spacing in milliseconds (recommend ~10x typical TPOT).
        instance_filter: If set, only count requests on these instance_ids.

    Returns:
        DataFrame with columns: t_s, concurrent_decode
    """
    if grid_ms <= 0:
        raise ValueError(f"grid_ms must be positive, got {grid_ms}")

    work = df.copy()
    if instance_filter is not None:
        work = work[work["instance id"].isin(instance_filter)]
        logger.info("Filtered to instances %s: %d rows", instance_filter, len(work))

    # Drop rows with no decode phase
    work = work[work["L_decode"] > 0]
    if len(work) == 0:
        raise ValueError("No rows with L_decode > 0 after filtering")

    work["decode_start"] = work["arrival"] + work["TTFT"]
    work["decode_end"] = work["end_time"]

    # Defensive: decode_end should be >= decode_start
    bad = work["decode_end"] < work["decode_start"]
    if bad.any():
        logger.warning(
            "%d rows have decode_end < decode_start; dropping", int(bad.sum())
        )
        work = work[~bad]

    grid_ns = int(grid_ms * NS_PER_MS)
    t_min = int(work["decode_start"].min())
    t_max = int(work["decode_end"].max())

    # Snap t_min down to a grid boundary so plots align across runs
    t_min_aligned = (t_min // grid_ns) * grid_ns
    grid = np.arange(t_min_aligned, t_max + grid_ns, grid_ns, dtype=np.int64)

    # Sweep-line via searchsorted: for each interval, increment the slice
    # [lo, hi) where grid points fall inside. Using side='left' for lo and
    # side='right' for hi makes the interval semantics half-open [start, end).
    counts = np.zeros(len(grid), dtype=np.int64)
    starts = work["decode_start"].to_numpy(dtype=np.int64)
    ends = work["decode_end"].to_numpy(dtype=np.int64)
    for s, e in zip(starts, ends):
        lo = np.searchsorted(grid, s, side="left")
        hi = np.searchsorted(grid, e, side="right")
        if hi > lo:
            counts[lo:hi] += 1

    ts = pd.DataFrame(
        {
            "t_s": grid / NS_PER_S,
            "concurrent_decode": counts,
        }
    )
    logger.info(
        "Timeseries: %d points, peak=%d, mean=%.2f",
        len(ts),
        ts["concurrent_decode"].max(),
        ts["concurrent_decode"].mean(),
    )
    return ts


# ---------------------------------------------------------------------------
# Steady-state window detection
# ---------------------------------------------------------------------------


@dataclass
class SteadyWindow:
    t_start_s: float
    t_end_s: float
    duration_s: float
    threshold: float
    peak: int
    mean_concurrent: float


def detect_steady_window(
    ts: pd.DataFrame,
    *,
    theta_ratio: float,
    hysteresis_s: float,
) -> Optional[SteadyWindow]:
    """Find the longest contiguous run where concurrent_decode >= threshold.

    threshold = peak * theta_ratio.

    Hysteresis: short dips below threshold (shorter than hysteresis_s) are
    treated as still being in steady state. This absorbs ASTRA-Sim transient
    stalls that would otherwise fragment the window.

    Returns None if no run is long enough to be meaningful (we use a minimum
    of 2 * hysteresis_s as the "meaningful" floor, falling back to 1s if
    hysteresis is zero).
    """
    if not (0 < theta_ratio <= 1):
        raise ValueError(f"theta_ratio must be in (0, 1], got {theta_ratio}")
    if hysteresis_s < 0:
        raise ValueError(f"hysteresis_s must be >= 0, got {hysteresis_s}")
    if len(ts) < 2:
        logger.warning("Timeseries too short (%d points)", len(ts))
        return None

    peak = int(ts["concurrent_decode"].max())
    if peak == 0:
        logger.warning("Peak concurrent decode is 0; nothing to analyze")
        return None

    threshold = peak * theta_ratio
    above = ts["concurrent_decode"].to_numpy() >= threshold

    grid_s = ts["t_s"].to_numpy()
    dt = float(grid_s[1] - grid_s[0])
    if dt <= 0:
        raise ValueError(f"Non-positive grid spacing: {dt}")

    # 임시: hysteresis가 0이면 morphological closing 생략
    if hysteresis_s > 0:
        closing_steps = max(1, int(round(hysteresis_s / dt)))
        above = _morphological_closing(above, closing_steps)

    # 가장 긴 연속 True 구간 찾기 (단순 스캔)
    best_lo, best_hi, best_len = -1, -1, 0
    cur_lo = -1
    for i, a in enumerate(above):
        if a and cur_lo < 0:
            cur_lo = i
        elif not a and cur_lo >= 0:
            run_len = i - cur_lo
            if run_len > best_len:
                best_len = run_len
                best_lo, best_hi = cur_lo, i
            cur_lo = -1
    # tail: above ends without ever going False
    if cur_lo >= 0:
        run_len = len(above) - cur_lo
        if run_len > best_len:
            best_len = run_len
            best_lo, best_hi = cur_lo, len(above)

    if best_lo < 0:
        logger.warning("No steady-state interval found above threshold=%.2f", threshold)
        return None

    # Reject windows that are too short to be statistically meaningful
    duration_s = (best_hi - best_lo) * dt
    min_meaningful = max(2 * hysteresis_s, 1.0)
    if duration_s < min_meaningful:
        logger.warning(
            "Detected window too short: %.3fs (min %.3fs). "
            "100-req trace is likely insufficient — consider rerunning with "
            "more requests.",
            duration_s,
            min_meaningful,
        )
        # Still return it; the caller can decide whether to use it.

    t_start_s = float(grid_s[best_lo])
    # use the last index inside the run (best_hi is exclusive)
    t_end_s = float(grid_s[min(best_hi - 1, len(grid_s) - 1)])
    in_window = ts.iloc[best_lo:best_hi]
    mean_concurrent = float(in_window["concurrent_decode"].mean())

    win = SteadyWindow(
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        duration_s=duration_s,
        threshold=threshold,
        peak=peak,
        mean_concurrent=mean_concurrent,
    )
    logger.info(
        "Steady window: [%.3fs, %.3fs] (%.3fs, threshold=%.2f, " "mean=%.2f, peak=%d)",
        win.t_start_s,
        win.t_end_s,
        win.duration_s,
        win.threshold,
        win.mean_concurrent,
        win.peak,
    )
    return win


def _morphological_closing(mask: np.ndarray, n_steps: int) -> np.ndarray:
    """Fill gaps in the True-mask shorter than n_steps.

    Equivalent to a 1D morphological closing with a flat structuring element
    of width n_steps. Used to absorb short transient dips so the steady-state
    window isn't fragmented by single-step noise.
    """
    if n_steps <= 1 or mask.size == 0:
        return mask
    out = mask.copy()
    n = len(out)
    i = 0
    while i < n:
        if out[i]:
            i += 1
            continue
        # find run of False
        j = i
        while j < n and not out[j]:
            j += 1
        gap_len = j - i
        # close if both ends are True (interior gap) and gap < n_steps
        if i > 0 and j < n and gap_len < n_steps:
            out[i:j] = True
        i = j
    return out


# ---------------------------------------------------------------------------
# Window metrics
# ---------------------------------------------------------------------------


@dataclass
class SteadyMetrics:
    window: SteadyWindow
    decode_tok_per_s: float
    n_full_requests: int
    n_partial_requests: int
    tpot_mean_ms: Optional[float]
    tpot_p50_ms: Optional[float]
    tpot_p99_ms: Optional[float]


def compute_steady_metrics(
    events: pd.DataFrame,
    df: pd.DataFrame,
    window: SteadyWindow,
    *,
    instance_filter: Optional[list[int]] = None,
) -> SteadyMetrics:
    """Aggregate metrics restricted to the steady-state window.

    decode_tok_per_s: counts token events whose t_ns lies in [t_start, t_end].
                     This is the most direct decode-throughput measurement.

    TPOT statistics: computed only over requests that both started decode
                    AND ended within the window (full lifecycle inside).
                    Partial-overlap requests are counted but excluded from
                    TPOT to avoid mixing in cooldown-dominated samples.
    """
    t_lo_ns = int(window.t_start_s * NS_PER_S)
    t_hi_ns = int(window.t_end_s * NS_PER_S)

    ev = events
    work = df
    if instance_filter is not None:
        ev = ev[ev["instance_id"].isin(instance_filter)]
        work = work[work["instance id"].isin(instance_filter)]

    # Token events strictly inside the window
    in_window = ev[(ev["t_ns"] >= t_lo_ns) & (ev["t_ns"] <= t_hi_ns)]
    duration_s = window.duration_s
    decode_tps = len(in_window) / duration_s if duration_s > 0 else 0.0

    # Per-request fully inside window
    work = work.copy()
    work["decode_start"] = work["arrival"] + work["TTFT"]
    work["decode_end"] = work["end_time"]
    fully_in = (
        (work["decode_start"] >= t_lo_ns)
        & (work["decode_end"] <= t_hi_ns)
        & (work["L_decode"] > 1)  # TPOT requires >=2 decode tokens
    )
    partial = (
        (work["decode_start"] <= t_hi_ns) & (work["decode_end"] >= t_lo_ns) & ~fully_in
    )

    full_df = work[fully_in]
    n_full = int(len(full_df))
    n_partial = int(partial.sum())

    if n_full > 0:
        tpot_ns = full_df["TPOT"].astype("int64")
        tpot_mean = float(tpot_ns.mean()) / NS_PER_MS
        tpot_p50 = float(tpot_ns.quantile(0.50)) / NS_PER_MS
        tpot_p99 = float(tpot_ns.quantile(0.99)) / NS_PER_MS
    else:
        tpot_mean = tpot_p50 = tpot_p99 = None
        logger.warning(
            "No requests fully inside the steady-state window (%d partial). "
            "TPOT stats unavailable. This is the classic 'window too short' "
            "symptom.",
            n_partial,
        )

    return SteadyMetrics(
        window=window,
        decode_tok_per_s=decode_tps,
        n_full_requests=n_full,
        n_partial_requests=n_partial,
        tpot_mean_ms=tpot_mean,
        tpot_p50_ms=tpot_p50,
        tpot_p99_ms=tpot_p99,
    )


# ---------------------------------------------------------------------------
# Plotting (optional)
# ---------------------------------------------------------------------------


def plot_diagnostic(
    ts: pd.DataFrame,
    window: Optional[SteadyWindow],
    out_path: Path,
) -> None:
    """Save a diagnostic plot: concurrent_decode vs time, with window shaded.

    Imports matplotlib lazily so the script runs without it when --plot is off.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts["t_s"], ts["concurrent_decode"], lw=1.0, label="concurrent decode")
    if window is not None:
        ax.axhline(
            window.threshold,
            color="red",
            ls="--",
            lw=0.8,
            label=f"threshold = {window.threshold:.1f}",
        )
        ax.axvspan(
            window.t_start_s,
            window.t_end_s,
            color="green",
            alpha=0.15,
            label=f"steady window ({window.duration_s:.2f}s)",
        )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("# concurrent decode requests")
    ax.set_title(f"Concurrent decode timeseries ({out_path.stem})")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Steady-state analysis for v0 simulator output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("input_csv", type=Path, help="Path to v0 per-request CSV.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: alongside input_csv).",
    )
    p.add_argument(
        "--grid-ms",
        type=float,
        default=100.0,
        help="Timeseries grid spacing in milliseconds. "
        "Rule of thumb: ~10x typical TPOT.",
    )
    p.add_argument(
        "--theta",
        type=float,
        default=0.9,
        help="Steady-state threshold as a fraction of peak " "concurrent decode.",
    )
    p.add_argument(
        "--hysteresis-s",
        type=float,
        default=2.0,
        help="Short dips below threshold up to this duration are "
        "absorbed (morphological closing).",
    )
    p.add_argument(
        "--instance-filter",
        type=int,
        nargs="+",
        default=None,
        help="Restrict analysis to these instance ids (useful for "
        "PD-disagg: pass only the decode-pool instance ids).",
    )
    p.add_argument(
        "--no-strict",
        action="store_true",
        help="Do not fail on ITL invariant mismatches " "(only warn).",
    )
    p.add_argument(
        "--plot",
        action="store_true",
        help="Save a diagnostic PNG of the concurrent-decode " "timeseries.",
    )
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logger(args.verbose)

    out_dir = args.out_dir if args.out_dir is not None else args.input_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.input_csv.stem

    # 1. Load and parse
    df_raw = load_v0_csv(args.input_csv)
    df = parse_itl_lists(df_raw)

    # 2. Self-check
    self_check = run_self_check(df, strict=not args.no_strict)

    # 3. Expand events and build timeseries
    events = expand_decode_events(df)
    ts = build_concurrent_decode_ts(
        df,
        grid_ms=args.grid_ms,
        instance_filter=args.instance_filter,
    )

    # 4. Detect window and compute metrics
    window = detect_steady_window(
        ts,
        theta_ratio=args.theta,
        hysteresis_s=args.hysteresis_s,
    )

    metrics: Optional[SteadyMetrics] = None
    if window is not None:
        metrics = compute_steady_metrics(
            events,
            df,
            window,
            instance_filter=args.instance_filter,
        )

    # 5. Persist outputs
    ts_path = out_dir / f"{stem}_concurrent_decode.csv"
    ts.to_csv(ts_path, index=False)
    logger.info("Saved timeseries: %s", ts_path)

    events_path = out_dir / f"{stem}_decode_events.csv"
    events.to_csv(events_path, index=False)
    logger.info("Saved decode events: %s", events_path)

    summary = {
        "input_csv": str(args.input_csv),
        "params": {
            "grid_ms": args.grid_ms,
            "theta": args.theta,
            "hysteresis_s": args.hysteresis_s,
            "instance_filter": args.instance_filter,
            "strict": not args.no_strict,
        },
        "self_check": asdict(self_check),
        "window": asdict(window) if window is not None else None,
        "metrics": (
            {
                "decode_tok_per_s": metrics.decode_tok_per_s,
                "n_full_requests": metrics.n_full_requests,
                "n_partial_requests": metrics.n_partial_requests,
                "tpot_mean_ms": metrics.tpot_mean_ms,
                "tpot_p50_ms": metrics.tpot_p50_ms,
                "tpot_p99_ms": metrics.tpot_p99_ms,
            }
            if metrics is not None
            else None
        ),
    }
    summary_path = out_dir / f"{stem}_steady_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary: %s", summary_path)

    # 6. Optional plot
    if args.plot:
        plot_path = out_dir / f"{stem}_concurrent_decode.png"
        plot_diagnostic(ts, window, plot_path)

    # 7. Console summary 출력 (디버깅용, 최종에는 JSON만 봐도 됨)
    print()
    print("=" * 60)
    print(f"Input: {args.input_csv}")
    print(
        f"Rows: {self_check.n_rows} | "
        f"Empty ITL: {self_check.n_empty_itl} | "
        f"ITL[0]==0: {self_check.n_zero_first_itl}"
    )
    if window is not None:
        print(
            f"Steady window: [{window.t_start_s:.3f}s, "
            f"{window.t_end_s:.3f}s] = {window.duration_s:.3f}s"
        )
        print(
            f"Peak={window.peak}, threshold={window.threshold:.2f}, "
            f"mean={window.mean_concurrent:.2f}"
        )
        if metrics is not None:
            print(f"Decode throughput: {metrics.decode_tok_per_s:.1f} tok/s")
            print(
                f"Full-in-window requests: {metrics.n_full_requests} "
                f"(partial: {metrics.n_partial_requests})"
            )
            if metrics.tpot_mean_ms is not None:
                print(
                    f"TPOT: mean={metrics.tpot_mean_ms:.2f}ms, "
                    f"p50={metrics.tpot_p50_ms:.2f}ms, "
                    f"p99={metrics.tpot_p99_ms:.2f}ms"
                )
    else:
        print("No steady-state window detected.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
