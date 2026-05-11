"""
Steady-state analysis for LLMServingSim using P1 timeseries CSV.

Reads the per-iteration timeseries written by the simulator (P1 patch),
filters to the decode pool by pd_type, detects the longest steady-state
window where concurrent decoders are near peak, and reports throughput /
TPOT metrics inside that window.

Compared to the v0 ITL-reconstruction script (analyze_steady_state_v0.py):
  - Input is the simulator's direct timeseries (1s grid, no reconstruction)
  - pd_type column resolves the instance-id ambiguity at source
  - Faster (no per-token expansion of 10k+ events)
  - Optional cross-validation mode reads BOTH inputs and compares peaks

Usage:
    # Standard: analyze a P1 timeseries
    python analyze_steady_state.py <timeseries_csv> [--per-request-csv <csv>]
                                                    [--theta 0.9]
                                                    [--hysteresis-s 2.0]
                                                    [--plot] [--verbose]

    # Cross-validation: compare timeseries-direct vs ITL-reconstructed
    python analyze_steady_state.py <timeseries_csv> --per-request-csv <csv>
                                                    --cross-validate
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

# Schema written by the P1 patch in serving/__main__.py
TIMESERIES_COLS = [
    "time_ns",
    "time_s",
    "instance_id",
    "pd_type",
    "node_id",
    "running_total",
    "running_prefill",
    "running_decode",
    "waiting",
    "npu_used_mb",
    "npu_util_pct",
]

# Per-request CSV columns (subset we use here)
PER_REQUEST_COLS = [
    "instance id",
    "request id",
    "input",
    "output",
    "arrival",
    "end_time",
    "latency",
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
# Loading
# ---------------------------------------------------------------------------


def load_timeseries_csv(path: Path) -> pd.DataFrame:
    """Load the P1 timeseries CSV with schema validation."""
    if not path.exists():
        raise FileNotFoundError(f"Timeseries CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = [c for c in TIMESERIES_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}\n"
            f"Found: {list(df.columns)}\n"
            f"Hint: this script expects the P1 timeseries format, not v0 "
            f"per-request CSV."
        )

    int_cols = [
        "time_ns",
        "instance_id",
        "node_id",
        "running_total",
        "running_prefill",
        "running_decode",
        "waiting",
    ]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")

    logger.info("Loaded %d timeseries rows from %s", len(df), path)
    return df


def load_per_request_csv(path: Path) -> pd.DataFrame:
    """Load the per-request CSV (optional input for TPOT distribution)."""
    if not path.exists():
        raise FileNotFoundError(f"Per-request CSV not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()

    missing = [c for c in PER_REQUEST_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns in {path}: {missing}\n"
            f"Found: {list(df.columns)}"
        )

    int_cols = [
        "instance id",
        "request id",
        "input",
        "output",
        "arrival",
        "end_time",
        "latency",
        "TTFT",
        "TPOT",
    ]
    for c in int_cols:
        df[c] = pd.to_numeric(df[c], errors="raise").astype("int64")

    logger.info("Loaded %d per-request rows from %s", len(df), path)
    return df


# ---------------------------------------------------------------------------
# Decode-pool aggregation
# ---------------------------------------------------------------------------


@dataclass
class PoolSummary:
    """Summary of decode pool composition."""

    pd_types_seen: list[str]
    n_decode_instances: int
    decode_instance_ids: list[int]
    n_total_instances: int


def summarize_pools(ts: pd.DataFrame) -> PoolSummary:
    """Inspect pd_type distribution and identify decode-pool instances."""
    pd_types = sorted(ts["pd_type"].unique().tolist())
    decode_mask = ts["pd_type"] == "decode"
    decode_inst_ids = sorted(ts.loc[decode_mask, "instance_id"].unique().tolist())
    all_inst_ids = sorted(ts["instance_id"].unique().tolist())

    summary = PoolSummary(
        pd_types_seen=pd_types,
        n_decode_instances=len(decode_inst_ids),
        decode_instance_ids=[int(x) for x in decode_inst_ids],
        n_total_instances=len(all_inst_ids),
    )
    logger.info(
        "Pool summary: pd_types=%s, decode_instances=%s, total=%d",
        summary.pd_types_seen,
        summary.decode_instance_ids,
        summary.n_total_instances,
    )

    if summary.n_decode_instances == 0:
        logger.warning(
            "No instances with pd_type='decode'. This timeseries may be "
            "from a unified (non-disaggregated) configuration, or pd_type "
            "is misconfigured. Falling back to all instances."
        )

    return summary


def aggregate_concurrent_decode(
    ts: pd.DataFrame,
    *,
    pool_filter: Optional[list[int]] = None,
) -> pd.DataFrame:
    """Aggregate running_decode across decode-pool instances per time step.

    If pool_filter is None, uses all instances with pd_type='decode'. If no
    such instances exist, falls back to all instances (with a warning).

    Returns:
        DataFrame with columns: t_s, concurrent_decode, total_waiting
        Each row is one time step (typically 1s apart per --log-interval).
    """
    work = ts.copy()

    if pool_filter is None:
        decode_inst = sorted(
            work.loc[work["pd_type"] == "decode", "instance_id"].unique()
        )
        if len(decode_inst) > 0:
            pool_filter = [int(x) for x in decode_inst]
            logger.info("Auto-selected decode pool instances: %s", pool_filter)
        else:
            logger.warning("Falling back to all instances (no pd_type='decode')")
            pool_filter = sorted(work["instance_id"].unique().tolist())

    work = work[work["instance_id"].isin(pool_filter)]
    if len(work) == 0:
        raise ValueError(f"No timeseries rows match pool_filter={pool_filter}")

    # Sum running_decode and waiting across pool instances at each time step.
    # time_ns is the natural grouping key (heartbeat ticks).
    agg = (
        work.groupby("time_ns", as_index=False)
        .agg(
            concurrent_decode=("running_decode", "sum"),
            total_waiting=("waiting", "sum"),
        )
        .sort_values("time_ns")
        .reset_index(drop=True)
    )
    agg["t_s"] = agg["time_ns"] / NS_PER_S
    agg = agg[["t_s", "concurrent_decode", "total_waiting"]]

    logger.info(
        "Aggregated timeseries: %d points, peak=%d, mean=%.2f, max_waiting=%d",
        len(agg),
        agg["concurrent_decode"].max(),
        agg["concurrent_decode"].mean(),
        agg["total_waiting"].max(),
    )
    return agg


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
    mean_waiting: float


def detect_steady_window(
    ts: pd.DataFrame,
    *,
    theta_ratio: float,
    hysteresis_s: float,
) -> Optional[SteadyWindow]:
    """Longest contiguous window where concurrent_decode >= peak * theta_ratio.

    Same algorithm as the v0 script. Hysteresis absorbs short transient dips
    (ASTRA-Sim stalls or batch-end / batch-start gaps) so the window isn't
    fragmented by single-step noise.
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
    if len(grid_s) < 2:
        return None
    dt = float(grid_s[1] - grid_s[0])
    if dt <= 0:
        raise ValueError(f"Non-positive grid spacing: {dt}")

    # 임시 디버깅: hysteresis=0이면 closing skip
    if hysteresis_s > 0:
        closing_steps = max(1, int(round(hysteresis_s / dt)))
        above = _morphological_closing(above, closing_steps)

    # Scan for the longest contiguous True run.
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
    if cur_lo >= 0:
        run_len = len(above) - cur_lo
        if run_len > best_len:
            best_len = run_len
            best_lo, best_hi = cur_lo, len(above)

    if best_lo < 0:
        logger.warning("No steady-state interval found above threshold=%.2f", threshold)
        return None

    duration_s = (best_hi - best_lo) * dt
    min_meaningful = max(2 * hysteresis_s, 1.0)
    if duration_s < min_meaningful:
        logger.warning(
            "Detected window too short: %.3fs (min %.3fs). Consider "
            "increasing num-reqs or compressing arrival rate.",
            duration_s,
            min_meaningful,
        )

    t_start_s = float(grid_s[best_lo])
    t_end_s = float(grid_s[min(best_hi - 1, len(grid_s) - 1)])
    in_window = ts.iloc[best_lo:best_hi]
    mean_concurrent = float(in_window["concurrent_decode"].mean())
    mean_waiting = float(in_window["total_waiting"].mean())

    win = SteadyWindow(
        t_start_s=t_start_s,
        t_end_s=t_end_s,
        duration_s=duration_s,
        threshold=threshold,
        peak=peak,
        mean_concurrent=mean_concurrent,
        mean_waiting=mean_waiting,
    )
    logger.info(
        "Steady window: [%.3fs, %.3fs] (%.3fs, threshold=%.2f, peak=%d, "
        "mean_concurrent=%.2f, mean_waiting=%.2f)",
        win.t_start_s,
        win.t_end_s,
        win.duration_s,
        win.threshold,
        win.peak,
        win.mean_concurrent,
        win.mean_waiting,
    )
    return win


def _morphological_closing(mask: np.ndarray, n_steps: int) -> np.ndarray:
    """Fill short False-gaps in the True-mask (gaps < n_steps).

    Reused verbatim from the v0 script. Equivalent to 1D morphological closing
    with a flat structuring element of width n_steps.
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
        j = i
        while j < n and not out[j]:
            j += 1
        gap_len = j - i
        if i > 0 and j < n and gap_len < n_steps:
            out[i:j] = True
        i = j
    return out


# ---------------------------------------------------------------------------
# Window metrics from per-request CSV
# ---------------------------------------------------------------------------


@dataclass
class SteadyMetrics:
    window: SteadyWindow
    decode_tok_per_s_avg: float  # avg(concurrent_decode) / 1s * window_s
    n_full_requests: int
    n_partial_requests: int
    tpot_mean_ms: Optional[float]
    tpot_p50_ms: Optional[float]
    tpot_p99_ms: Optional[float]
    ttft_mean_ms: Optional[float]
    ttft_p99_ms: Optional[float]


def compute_steady_metrics(
    timeseries_agg: pd.DataFrame,
    per_request: Optional[pd.DataFrame],
    window: SteadyWindow,
) -> SteadyMetrics:
    """Compute steady-state metrics inside the detected window.

    Throughput estimate uses the timeseries directly: average concurrent
    decoders inside the window times one-token-per-second-per-slot is a
    coarse proxy. A more accurate token-rate would require per-token
    timestamps (see v0 script for that path).

    TPOT/TTFT are computed from the per-request CSV when available,
    restricted to requests that both started and ended inside the window
    (to avoid mixing cooldown samples). Requires per-request input.
    """
    t_lo_ns = int(window.t_start_s * NS_PER_S)
    t_hi_ns = int(window.t_end_s * NS_PER_S)

    # Throughput proxy: average concurrent decoders in window.
    # Each decoder emits ~1 token / TPOT_s. Without per-token timestamps
    # we report the average concurrency itself; multiply by 1/TPOT to get
    # tokens/s once TPOT is known.
    in_win_ts = timeseries_agg[
        (timeseries_agg["t_s"] * NS_PER_S >= t_lo_ns)
        & (timeseries_agg["t_s"] * NS_PER_S <= t_hi_ns)
    ]
    avg_concurrent = (
        float(in_win_ts["concurrent_decode"].mean()) if len(in_win_ts) > 0 else 0.0
    )

    n_full = 0
    n_partial = 0
    tpot_mean = tpot_p50 = tpot_p99 = None
    ttft_mean = ttft_p99 = None
    throughput_proxy = 0.0

    if per_request is not None:
        work = per_request.copy()
        work["decode_start"] = work["arrival"] + work["TTFT"]
        work["decode_end"] = work["end_time"]
        # In the P1 per-request CSV, the "output" column records the number
        # of generated (decode) tokens only — NOT input + decode. So
        # L_decode is exactly output.
        work["L_decode"] = work["output"]

        fully_in = (
            (work["decode_start"] >= t_lo_ns)
            & (work["decode_end"] <= t_hi_ns)
            & (work["L_decode"] > 1)
        )
        partial = (
            (work["decode_start"] <= t_hi_ns)
            & (work["decode_end"] >= t_lo_ns)
            & ~fully_in
        )
        full_df = work[fully_in]
        n_full = int(len(full_df))
        n_partial = int(partial.sum())

        if n_full > 0:
            tpot_ns = full_df["TPOT"].astype("int64")
            ttft_ns = full_df["TTFT"].astype("int64")
            tpot_mean = float(tpot_ns.mean()) / NS_PER_MS
            tpot_p50 = float(tpot_ns.quantile(0.50)) / NS_PER_MS
            tpot_p99 = float(tpot_ns.quantile(0.99)) / NS_PER_MS
            ttft_mean = float(ttft_ns.mean()) / NS_PER_MS
            ttft_p99 = float(ttft_ns.quantile(0.99)) / NS_PER_MS

            # Throughput: avg_concurrent * (1 token / TPOT) is the
            # decode-side token rate when the pool runs at avg_concurrent.
            tpot_s = tpot_mean / 1000.0
            if tpot_s > 0:
                throughput_proxy = avg_concurrent / tpot_s
        else:
            logger.warning(
                "No requests fully inside window (%d partial). TPOT stats "
                "unavailable. Window is too short relative to per-request "
                "decode duration.",
                n_partial,
            )

    return SteadyMetrics(
        window=window,
        decode_tok_per_s_avg=throughput_proxy,
        n_full_requests=n_full,
        n_partial_requests=n_partial,
        tpot_mean_ms=tpot_mean,
        tpot_p50_ms=tpot_p50,
        tpot_p99_ms=tpot_p99,
        ttft_mean_ms=ttft_mean,
        ttft_p99_ms=ttft_p99,
    )


# ---------------------------------------------------------------------------
# Cross-validation against v0 ITL reconstruction
# ---------------------------------------------------------------------------


@dataclass
class CrossValidationReport:
    """Compare timeseries-direct vs ITL-reconstructed concurrent decode."""

    ts_peak: int
    reconstructed_peak: int
    peak_diff_pct: float
    ts_mean: float
    reconstructed_mean: float
    mean_diff_pct: float
    notes: list[str]


def cross_validate(
    timeseries_agg: pd.DataFrame,
    per_request: pd.DataFrame,
    *,
    decode_instance_ids: Optional[list[int]] = None,
    grid_ms: float = 1000.0,
) -> CrossValidationReport:
    """Build the v0-style reconstructed concurrent_decode and compare.

    The reconstruction uses each request's [arrival+TTFT, end_time] as the
    decode interval and counts overlaps on the same time grid as the
    timeseries. The two should be close in peak and mean if both
    implementations are correct.
    """
    notes: list[str] = []

    work = per_request.copy()
    if decode_instance_ids is not None:
        before = len(work)
        work = work[work["instance id"].isin(decode_instance_ids)]
        notes.append(
            f"Filtered per-request to decode instance(s) {decode_instance_ids}: "
            f"{before} -> {len(work)} rows"
        )

    work["decode_start"] = work["arrival"] + work["TTFT"]
    work["decode_end"] = work["end_time"]
    # See compute_steady_metrics: output = decode tokens only.
    work["L_decode"] = work["output"]
    work = work[work["L_decode"] > 0]
    if len(work) == 0:
        raise ValueError("No rows with L_decode > 0 after filtering")

    grid_ns = int(grid_ms * NS_PER_MS)
    t_min = int(work["decode_start"].min())
    t_max = int(work["decode_end"].max())
    t_min_aligned = (t_min // grid_ns) * grid_ns
    grid = np.arange(t_min_aligned, t_max + grid_ns, grid_ns, dtype=np.int64)

    counts = np.zeros(len(grid), dtype=np.int64)
    starts = work["decode_start"].to_numpy(dtype=np.int64)
    ends = work["decode_end"].to_numpy(dtype=np.int64)
    for s, e in zip(starts, ends):
        lo = np.searchsorted(grid, s, side="left")
        hi = np.searchsorted(grid, e, side="right")
        if hi > lo:
            counts[lo:hi] += 1

    rc_peak = int(counts.max())
    rc_mean = float(counts.mean())

    ts_peak = int(timeseries_agg["concurrent_decode"].max())
    ts_mean = float(timeseries_agg["concurrent_decode"].mean())

    def _pct_diff(a: float, b: float) -> float:
        # symmetric percent difference; 0 if both zero
        if a == 0 and b == 0:
            return 0.0
        return 200.0 * abs(a - b) / (abs(a) + abs(b))

    report = CrossValidationReport(
        ts_peak=ts_peak,
        reconstructed_peak=rc_peak,
        peak_diff_pct=_pct_diff(ts_peak, rc_peak),
        ts_mean=ts_mean,
        reconstructed_mean=rc_mean,
        mean_diff_pct=_pct_diff(ts_mean, rc_mean),
        notes=notes,
    )

    logger.info(
        "Cross-validation: peak ts=%d vs reconstructed=%d (diff=%.1f%%), "
        "mean ts=%.2f vs reconstructed=%.2f (diff=%.1f%%)",
        report.ts_peak,
        report.reconstructed_peak,
        report.peak_diff_pct,
        report.ts_mean,
        report.reconstructed_mean,
        report.mean_diff_pct,
    )
    return report


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_diagnostic(
    ts: pd.DataFrame,
    window: Optional[SteadyWindow],
    out_path: Path,
) -> None:
    """Save a diagnostic plot: concurrent_decode + waiting + steady window."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available; skipping plot")
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(
        ts["t_s"],
        ts["concurrent_decode"],
        lw=1.2,
        label="concurrent decode",
        color="steelblue",
    )
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("# concurrent decode requests", color="steelblue")
    ax1.tick_params(axis="y", labelcolor="steelblue")

    ax2 = ax1.twinx()
    ax2.plot(
        ts["t_s"],
        ts["total_waiting"],
        lw=0.8,
        label="waiting queue",
        color="darkorange",
        alpha=0.6,
    )
    ax2.set_ylabel("# waiting requests", color="darkorange")
    ax2.tick_params(axis="y", labelcolor="darkorange")

    if window is not None:
        ax1.axhline(
            window.threshold,
            color="red",
            ls="--",
            lw=0.8,
            label=f"threshold = {window.threshold:.1f}",
        )
        ax1.axvspan(
            window.t_start_s,
            window.t_end_s,
            color="green",
            alpha=0.15,
            label=f"steady ({window.duration_s:.2f}s)",
        )

    ax1.set_title(f"Decode pool timeseries ({out_path.stem})")
    ax1.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Saved plot: %s", out_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Steady-state analysis using P1 timeseries CSV.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "timeseries_csv",
        type=Path,
        help="Path to *_timeseries.csv emitted by the simulator.",
    )
    p.add_argument(
        "--per-request-csv",
        type=Path,
        default=None,
        help="Per-request CSV (same simulation, *.csv without "
        "_timeseries suffix). Required for TPOT/TTFT and "
        "cross-validation.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: alongside timeseries CSV).",
    )
    p.add_argument(
        "--instance-filter",
        type=int,
        nargs="+",
        default=None,
        help="Restrict to these instance ids. Default: auto-pick "
        "instances with pd_type='decode'.",
    )
    p.add_argument(
        "--theta",
        type=float,
        default=0.9,
        help="Steady-state threshold as fraction of peak.",
    )
    p.add_argument(
        "--hysteresis-s",
        type=float,
        default=2.0,
        help="Absorb dips below threshold up to this duration.",
    )
    p.add_argument(
        "--cross-validate",
        action="store_true",
        help="Compute v0-style ITL-reconstructed concurrent decode "
        "and compare to timeseries. Requires --per-request-csv.",
    )
    p.add_argument("--plot", action="store_true", help="Save a diagnostic PNG.")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG-level logging.")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_logger(args.verbose)

    if args.cross_validate and args.per_request_csv is None:
        logger.error("--cross-validate requires --per-request-csv")
        return 1

    out_dir = args.out_dir if args.out_dir is not None else args.timeseries_csv.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Strip _timeseries suffix for output stem so artifacts pair naturally
    # with the run name.
    stem = args.timeseries_csv.stem
    if stem.endswith("_timeseries"):
        stem = stem[: -len("_timeseries")]

    # 1. Load
    ts_raw = load_timeseries_csv(args.timeseries_csv)
    pool_summary = summarize_pools(ts_raw)

    per_request = None
    if args.per_request_csv is not None:
        per_request = load_per_request_csv(args.per_request_csv)

    # 2. Aggregate decode pool
    ts_agg = aggregate_concurrent_decode(
        ts_raw,
        pool_filter=args.instance_filter,
    )

    # 3. Detect window
    window = detect_steady_window(
        ts_agg,
        theta_ratio=args.theta,
        hysteresis_s=args.hysteresis_s,
    )

    metrics = None
    if window is not None:
        metrics = compute_steady_metrics(ts_agg, per_request, window)

    # 4. Optional cross-validation
    #
    # NOTE: per-request CSV's "instance id" column records the SCHEDULER
    # instance that handled the request — typically a prefill-pool instance
    # in PD-disagg setups. It is NOT the same identifier as the
    # cluster-config index used in timeseries' instance_id column.
    #
    # Since every completed request eventually decodes on the decode pool,
    # ALL per-request rows contribute to the decode-pool concurrency
    # reconstruction. We therefore do NOT filter by instance here.
    cv_report = None
    if args.cross_validate:
        if args.instance_filter is not None:
            # User explicitly asked to filter — honor it, but warn.
            logger.warning(
                "Cross-validation with --instance-filter is unusual. "
                "Per-request CSV instance ids may not match cluster config "
                "instance_ids. Consider running without --instance-filter "
                "for a fairer comparison."
            )
            cv_inst = args.instance_filter
        else:
            cv_inst = None  # use all rows
        cv_report = cross_validate(
            ts_agg,
            per_request,
            decode_instance_ids=cv_inst,
            grid_ms=1000.0,
        )

    # 5. Persist outputs
    ts_path = out_dir / f"{stem}_decode_pool_ts.csv"
    ts_agg.to_csv(ts_path, index=False)
    logger.info("Saved aggregated timeseries: %s", ts_path)

    summary = {
        "timeseries_csv": str(args.timeseries_csv),
        "per_request_csv": str(args.per_request_csv) if args.per_request_csv else None,
        "params": {
            "theta": args.theta,
            "hysteresis_s": args.hysteresis_s,
            "instance_filter": args.instance_filter,
        },
        "pool_summary": asdict(pool_summary),
        "window": asdict(window) if window is not None else None,
        "metrics": (
            {
                "decode_tok_per_s_avg": metrics.decode_tok_per_s_avg,
                "n_full_requests": metrics.n_full_requests,
                "n_partial_requests": metrics.n_partial_requests,
                "tpot_mean_ms": metrics.tpot_mean_ms,
                "tpot_p50_ms": metrics.tpot_p50_ms,
                "tpot_p99_ms": metrics.tpot_p99_ms,
                "ttft_mean_ms": metrics.ttft_mean_ms,
                "ttft_p99_ms": metrics.ttft_p99_ms,
            }
            if metrics is not None
            else None
        ),
        "cross_validation": asdict(cv_report) if cv_report is not None else None,
    }
    summary_path = out_dir / f"{stem}_steady_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Saved summary: %s", summary_path)

    # 6. Plot
    if args.plot:
        plot_path = out_dir / f"{stem}_decode_pool_ts.png"
        plot_diagnostic(ts_agg, window, plot_path)

    # 7. 콘솔 요약 (디버깅용, JSON으로 충분하면 삭제 가능)
    print()
    print("=" * 60)
    print(f"Input: {args.timeseries_csv}")
    print(
        f"Pool: {pool_summary.n_decode_instances} decode instance(s) "
        f"({pool_summary.decode_instance_ids}) of "
        f"{pool_summary.n_total_instances} total"
    )
    if window is not None:
        print(
            f"Steady window: [{window.t_start_s:.3f}s, "
            f"{window.t_end_s:.3f}s] = {window.duration_s:.3f}s"
        )
        print(
            f"  peak={window.peak}, threshold={window.threshold:.2f}, "
            f"mean_concurrent={window.mean_concurrent:.2f}, "
            f"mean_waiting={window.mean_waiting:.2f}"
        )
        if metrics is not None and metrics.tpot_mean_ms is not None:
            print(f"Throughput proxy: {metrics.decode_tok_per_s_avg:.1f} tok/s")
            print(
                f"Full-in-window: {metrics.n_full_requests} "
                f"(partial: {metrics.n_partial_requests})"
            )
            print(
                f"TPOT: mean={metrics.tpot_mean_ms:.2f}ms, "
                f"p50={metrics.tpot_p50_ms:.2f}ms, "
                f"p99={metrics.tpot_p99_ms:.2f}ms"
            )
            print(
                f"TTFT: mean={metrics.ttft_mean_ms:.2f}ms, "
                f"p99={metrics.ttft_p99_ms:.2f}ms"
            )
    else:
        print("No steady-state window detected.")

    if cv_report is not None:
        print()
        print("Cross-validation (timeseries-direct vs ITL-reconstructed):")
        print(
            f"  peak: ts={cv_report.ts_peak} vs recon={cv_report.reconstructed_peak}"
            f" ({cv_report.peak_diff_pct:.1f}% diff)"
        )
        print(
            f"  mean: ts={cv_report.ts_mean:.2f} vs recon={cv_report.reconstructed_mean:.2f}"
            f" ({cv_report.mean_diff_pct:.1f}% diff)"
        )
        if cv_report.peak_diff_pct < 20 and cv_report.mean_diff_pct < 20:
            print("  -> Methods agree within 20%. Analysis is validated.")
        else:
            print("  -> WARNING: methods disagree by >20%. Investigate.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
