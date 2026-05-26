"""plot_load_scale.py — compare steady-state metrics across load scaling factors.

Reads per-request CSVs and timeseries CSVs produced by run_load_scale.sh,
detects steady-state windows, and outputs a multi-panel comparison figure.

Usage (from repo root):
    python experiments/plot_load_scale.py \
        --outdir outputs/load_scale_exp \
        --scales 0.25 0.5 1.0 2.0 4.0 \
        --theta 0.8 --hysteresis-s 2.0 \
        --output outputs/load_scale_exp/comparison.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

NS_PER_S = 1_000_000_000
NS_PER_MS = 1_000_000


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def load_per_request(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in ["arrival", "end_time", "latency", "TTFT", "TPOT", "ITL"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_timeseries(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


# ---------------------------------------------------------------------------
# Steady-state detection (mirrors analyze_steady_state.py)
# ---------------------------------------------------------------------------

def _morphological_closing(mask: np.ndarray, n_steps: int) -> np.ndarray:
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


def aggregate_decode(ts: pd.DataFrame) -> pd.DataFrame:
    """Sum running_decode + waiting across all instances per time step."""
    agg = (
        ts.groupby("time_ns", as_index=False)
        .agg(
            concurrent_decode=("running_decode", "sum"),
            total_waiting=("waiting", "sum"),
        )
        .sort_values("time_ns")
        .reset_index(drop=True)
    )
    agg["t_s"] = agg["time_ns"] / NS_PER_S
    return agg


def detect_steady_window(agg: pd.DataFrame, theta: float, hysteresis_s: float):
    """Return (t_start_s, t_end_s, peak, mean_concurrent) or None."""
    if len(agg) < 2:
        return None
    peak = int(agg["concurrent_decode"].max())
    if peak == 0:
        return None
    threshold = peak * theta
    above = (agg["concurrent_decode"].to_numpy() >= threshold)
    grid_s = agg["t_s"].to_numpy()
    dt = float(grid_s[1] - grid_s[0])
    if dt <= 0:
        return None
    if hysteresis_s > 0:
        closing_steps = max(1, int(round(hysteresis_s / dt)))
        above = _morphological_closing(above, closing_steps)

    best_lo, best_hi, best_len = -1, -1, 0
    cur_lo = -1
    for i, a in enumerate(above):
        if a and cur_lo < 0:
            cur_lo = i
        elif not a and cur_lo >= 0:
            if (i - cur_lo) > best_len:
                best_len = i - cur_lo
                best_lo, best_hi = cur_lo, i
            cur_lo = -1
    if cur_lo >= 0:
        if (len(above) - cur_lo) > best_len:
            best_lo, best_hi = cur_lo, len(above)

    if best_lo < 0:
        return None
    return (
        float(grid_s[best_lo]),
        float(grid_s[min(best_hi - 1, len(grid_s) - 1)]),
        peak,
        float(agg["concurrent_decode"].iloc[best_lo:best_hi].mean()),
    )


def steady_metrics(per_req: pd.DataFrame, t_start_s: float, t_end_s: float) -> dict:
    """Extract TTFT / TPOT / latency for requests with decode period in window."""
    t_lo = int(t_start_s * NS_PER_S)
    t_hi = int(t_end_s * NS_PER_S)
    work = per_req.copy()
    work["decode_start"] = work["arrival"] + work["TTFT"]
    fully_in = (work["decode_start"] >= t_lo) & (work["end_time"] <= t_hi)
    in_win = work[fully_in]

    if len(in_win) == 0:
        # Fall back to all requests whose arrival falls in the window
        in_win = work[(work["arrival"] >= t_lo) & (work["arrival"] <= t_hi)]

    if len(in_win) == 0:
        return {}

    tpot_ms = in_win["TPOT"] / NS_PER_MS
    ttft_ms = in_win["TTFT"] / NS_PER_MS
    lat_ms  = in_win["latency"] / NS_PER_MS

    # Request-level throughput: completed requests / window duration
    duration_s = t_end_s - t_start_s
    req_throughput = len(in_win) / max(duration_s, 1.0)

    return {
        "n": len(in_win),
        "tpot_mean_ms":  float(tpot_ms.mean()),
        "tpot_p50_ms":   float(tpot_ms.quantile(0.50)),
        "tpot_p99_ms":   float(tpot_ms.quantile(0.99)),
        "ttft_mean_ms":  float(ttft_ms.mean()),
        "ttft_p99_ms":   float(ttft_ms.quantile(0.99)),
        "lat_mean_ms":   float(lat_ms.mean()),
        "lat_p99_ms":    float(lat_ms.quantile(0.99)),
        "req_throughput": req_throughput,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Load-scale comparison plot")
    p.add_argument("--outdir", type=Path, default=Path("outputs/load_scale_exp"))
    p.add_argument(
        "--scales", type=float, nargs="+",
        default=[0.25, 0.5, 1.0, 2.0, 4.0],
        help="Time-interval scaling factors to include (same as --load-scale flags used)",
    )
    p.add_argument("--theta", type=float, default=0.8,
                   help="Fraction of peak concurrency for steady-state threshold")
    p.add_argument("--hysteresis-s", type=float, default=2.0)
    p.add_argument("--output", type=Path, default=None,
                   help="Output PNG path (default: <outdir>/comparison.png)")
    return p.parse_args()


def main():
    args = parse_args()
    outdir: Path = args.outdir
    out_png = args.output or outdir / "comparison.png"
    scales = sorted(args.scales)

    # ------------------------------------------------------------------ load
    results = {}
    for sc in scales:
        tag = f"scale{sc}"
        req_csv  = outdir / f"{tag}.csv"
        ts_csv   = outdir / f"{tag}_timeseries.csv"
        per_req  = load_per_request(req_csv)
        ts_raw   = load_timeseries(ts_csv)

        if per_req is None:
            print(f"[WARN] missing {req_csv}, skipping scale={sc}", file=sys.stderr)
            continue

        ts_agg = None
        window = None
        metrics = {}
        if ts_raw is not None and len(ts_raw) > 0:
            ts_agg = aggregate_decode(ts_raw)
            win = detect_steady_window(ts_agg, args.theta, args.hysteresis_s)
            if win is not None:
                t_start, t_end, peak, mean_conc = win
                window = {"t_start": t_start, "t_end": t_end, "peak": peak,
                          "mean_concurrent": mean_conc}
                metrics = steady_metrics(per_req, t_start, t_end)

        if not metrics:
            # No steady window: use all completed requests
            tpot_ms = per_req["TPOT"] / NS_PER_MS
            ttft_ms = per_req["TTFT"] / NS_PER_MS
            lat_ms  = per_req["latency"] / NS_PER_MS
            dur_s   = (per_req["end_time"].max() - per_req["arrival"].min()) / NS_PER_S
            metrics = {
                "n": len(per_req),
                "tpot_mean_ms":   float(tpot_ms.mean()),
                "tpot_p50_ms":    float(tpot_ms.quantile(0.50)),
                "tpot_p99_ms":    float(tpot_ms.quantile(0.99)),
                "ttft_mean_ms":   float(ttft_ms.mean()),
                "ttft_p99_ms":    float(ttft_ms.quantile(0.99)),
                "lat_mean_ms":    float(lat_ms.mean()),
                "lat_p99_ms":     float(lat_ms.quantile(0.99)),
                "req_throughput": len(per_req) / max(dur_s, 1.0),
            }

        results[sc] = {
            "per_req": per_req,
            "ts_agg":  ts_agg,
            "window":  window,
            "metrics": metrics,
        }
        print(f"scale={sc}: n={metrics['n']}, TPOT_mean={metrics['tpot_mean_ms']:.1f}ms, "
              f"TTFT_mean={metrics['ttft_mean_ms']:.1f}ms, "
              f"req/s={metrics['req_throughput']:.2f}")

    if not results:
        print("No results found. Run experiments/run_load_scale.sh first.", file=sys.stderr)
        sys.exit(1)

    valid_scales = sorted(results.keys())
    # Effective load factor = 1 / scale  (scale=0.5 → 2x load)
    load_factors = [1.0 / s for s in valid_scales]

    # ------------------------------------------------------------------ save JSON summary
    summary_path = outdir / "comparison_summary.json"
    summary_out = {
        str(sc): {k: v for k, v in res["metrics"].items() if k != "n"}
        for sc, res in results.items()
    }
    with open(summary_path, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Summary JSON: {summary_path}")

    # ------------------------------------------------------------------ plot
    COLORS = plt.cm.plasma(np.linspace(0.15, 0.85, len(valid_scales)))

    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    ax_tpot   = fig.add_subplot(gs[0, 0])
    ax_ttft   = fig.add_subplot(gs[0, 1])
    ax_lat    = fig.add_subplot(gs[0, 2])
    ax_cdf    = fig.add_subplot(gs[1, :2])
    ax_ttft_cdf = fig.add_subplot(gs[1, 2])
    ax_ts     = fig.add_subplot(gs[2, :])

    # ---------- Panel 1: TPOT mean + p99 vs load factor ----------
    tpot_means = [results[s]["metrics"]["tpot_mean_ms"] for s in valid_scales]
    tpot_p99s  = [results[s]["metrics"]["tpot_p99_ms"]  for s in valid_scales]
    ax_tpot.plot(load_factors, tpot_means, "o-", color="steelblue", label="mean", lw=1.8)
    ax_tpot.plot(load_factors, tpot_p99s,  "s--", color="steelblue", label="p99", alpha=0.65, lw=1.5)
    ax_tpot.set_xscale("log")
    ax_tpot.set_xlabel("Load factor (1 / time-scale)")
    ax_tpot.set_ylabel("TPOT (ms)")
    ax_tpot.set_title("TPOT vs Load")
    ax_tpot.legend(fontsize=8)
    ax_tpot.grid(True, alpha=0.3)
    _annotate_scales(ax_tpot, load_factors, tpot_means, valid_scales)

    # ---------- Panel 2: TTFT mean + p99 vs load factor ----------
    ttft_means = [results[s]["metrics"]["ttft_mean_ms"] for s in valid_scales]
    ttft_p99s  = [results[s]["metrics"]["ttft_p99_ms"]  for s in valid_scales]
    ax_ttft.plot(load_factors, ttft_means, "o-", color="darkorange", label="mean", lw=1.8)
    ax_ttft.plot(load_factors, ttft_p99s,  "s--", color="darkorange", label="p99", alpha=0.65, lw=1.5)
    ax_ttft.set_xscale("log")
    ax_ttft.set_xlabel("Load factor (1 / time-scale)")
    ax_ttft.set_ylabel("TTFT (ms)")
    ax_ttft.set_title("TTFT vs Load")
    ax_ttft.legend(fontsize=8)
    ax_ttft.grid(True, alpha=0.3)
    _annotate_scales(ax_ttft, load_factors, ttft_means, valid_scales)

    # ---------- Panel 3: E2E latency vs load factor ----------
    lat_means = [results[s]["metrics"]["lat_mean_ms"] for s in valid_scales]
    lat_p99s  = [results[s]["metrics"]["lat_p99_ms"]  for s in valid_scales]
    ax_lat.plot(load_factors, lat_means, "o-", color="forestgreen", label="mean", lw=1.8)
    ax_lat.plot(load_factors, lat_p99s,  "s--", color="forestgreen", label="p99", alpha=0.65, lw=1.5)
    ax_lat.set_xscale("log")
    ax_lat.set_xlabel("Load factor (1 / time-scale)")
    ax_lat.set_ylabel("E2E Latency (ms)")
    ax_lat.set_title("E2E Latency vs Load")
    ax_lat.legend(fontsize=8)
    ax_lat.grid(True, alpha=0.3)
    _annotate_scales(ax_lat, load_factors, lat_means, valid_scales)

    # ---------- Panel 4: TPOT CDF across scales ----------
    for sc, color in zip(valid_scales, COLORS):
        tpot_vals = results[sc]["per_req"]["TPOT"] / NS_PER_MS
        tpot_sorted = np.sort(tpot_vals)
        cdf = np.arange(1, len(tpot_sorted) + 1) / len(tpot_sorted)
        lf = 1.0 / sc
        ax_cdf.plot(tpot_sorted, cdf, color=color,
                    label=f"scale={sc} (×{lf:.2f} load)", lw=1.5)
    ax_cdf.set_xlabel("TPOT (ms)")
    ax_cdf.set_ylabel("CDF")
    ax_cdf.set_title("TPOT CDF across load scales")
    ax_cdf.legend(fontsize=8)
    ax_cdf.grid(True, alpha=0.3)
    ax_cdf.set_xlim(left=0)

    # ---------- Panel 5: TTFT CDF ----------
    for sc, color in zip(valid_scales, COLORS):
        ttft_vals = results[sc]["per_req"]["TTFT"] / NS_PER_MS
        ttft_sorted = np.sort(ttft_vals)
        cdf = np.arange(1, len(ttft_sorted) + 1) / len(ttft_sorted)
        lf = 1.0 / sc
        ax_ttft_cdf.plot(ttft_sorted, cdf, color=color,
                         label=f"×{lf:.2f}", lw=1.5)
    ax_ttft_cdf.set_xlabel("TTFT (ms)")
    ax_ttft_cdf.set_ylabel("CDF")
    ax_ttft_cdf.set_title("TTFT CDF")
    ax_ttft_cdf.legend(fontsize=8)
    ax_ttft_cdf.grid(True, alpha=0.3)
    ax_ttft_cdf.set_xlim(left=0)

    # ---------- Panel 6: Timeseries concurrent decode ----------
    for sc, color in zip(valid_scales, COLORS):
        ts_agg = results[sc]["ts_agg"]
        if ts_agg is None:
            continue
        lf = 1.0 / sc
        ax_ts.plot(ts_agg["t_s"], ts_agg["concurrent_decode"],
                   color=color, lw=1.3,
                   label=f"scale={sc} (×{lf:.2f} load)")
        win = results[sc]["window"]
        if win is not None:
            ax_ts.axvspan(win["t_start"], win["t_end"], color=color, alpha=0.12)

    ax_ts.set_xlabel("Simulation time (s)")
    ax_ts.set_ylabel("# Concurrent requests")
    ax_ts.set_title("Concurrent requests over time (shaded = steady window)")
    ax_ts.legend(fontsize=8, loc="upper right")
    ax_ts.grid(True, alpha=0.3)

    # ---------- Main title ----------
    baseline_rate = 10.0  # SPS of the source workload
    fig.suptitle(
        f"Load Scaling Experiment — sharegpt-llama-3.1-8B, 300 req\n"
        f"Baseline: {baseline_rate} req/s  |  time-scale × factor = inter-arrival intervals",
        fontsize=12, y=1.01,
    )

    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"Saved: {out_png}")


def _annotate_scales(ax, load_factors, values, scales):
    for lf, val, sc in zip(load_factors, values, scales):
        ax.annotate(
            f"×{lf:.2f}",
            xy=(lf, val),
            xytext=(0, 6),
            textcoords="offset points",
            ha="center", fontsize=7, color="gray",
        )


if __name__ == "__main__":
    main()
