#!/bin/bash
#
# Analyze a main simulation run end-to-end and print every metric needed
# for the steady-state report.
#
# Usage:
#   bash outputs/analyze_main_run.sh [<run_name>]
#
# Default run name: custom_pdd_run_req_1000_sized10
#
# Expected files:
#   outputs/<run_name>.csv               — per-request CSV
#   outputs/<run_name>_timeseries.csv    — P1 timeseries CSV
#   outputs/<run_name>.log               — simulator stdout (from tee)
#
# Output:
#   outputs/<run_name>_decode_pool_ts.csv  — aggregated decode-pool timeseries
#   outputs/<run_name>_decode_pool_ts.png  — diagnostic plot
#   outputs/<run_name>_steady_summary.json — window + metrics summary
#
# All TBD values in the report draft can be read from this script's stdout.

set -e

RUN_NAME="${1:-custom_pdd_run_req_1000_sized10}"
TS_CSV="outputs/${RUN_NAME}_timeseries.csv"
PR_CSV="outputs/${RUN_NAME}.csv"
LOG="outputs/${RUN_NAME}.log"

# Sanity checks — fail fast with a clear message if expected files are missing.
for f in "$TS_CSV" "$PR_CSV"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] Missing required file: $f"
        echo "        Did the simulation finish? Check tmux session and logs."
        exit 1
    fi
done

echo "============================================================"
echo "Run name: $RUN_NAME"
echo "Per-request CSV: $PR_CSV ($(wc -l < "$PR_CSV") rows)"
echo "Timeseries CSV:  $TS_CSV ($(wc -l < "$TS_CSV") rows)"
echo "============================================================"
echo

# Step 1: Steady-state analysis.
echo "[1/4] Running steady-state analysis..."
echo "------------------------------------------------------------"
python outputs/analyze_steady_state.py \
    "$TS_CSV" \
    --per-request-csv "$PR_CSV" \
    --plot --verbose
echo

# Step 2: Naive whole-simulation averages from simulator stdout.
# These are the "before" values for the comparison table in the report.
echo "[2/4] Naive whole-simulation averages (for comparison):"
echo "------------------------------------------------------------"
if [ -f "$LOG" ]; then
    # The simulator prints these in a "Throughput Results" block — pull
    # the lines after the section header. Falling back to grep if the
    # awk-based extraction misses anything.
    grep -E "(Total simulation time|Total clocks|Total latency|Total requests|Total input tokens|Total generated tokens|Request throughput|Average prompt throughput|Average generation throughput|Total token throughput)" "$LOG" \
        || echo "[WARN] Could not find throughput summary lines in $LOG"
else
    echo "[WARN] No log file at $LOG — naive averages unavailable."
    echo "       (Was simulator stdout captured with 'tee'?)"
fi
echo

# Step 3: Steady-state summary JSON (pretty-printed for the report).
SUMMARY_JSON="outputs/${RUN_NAME}_steady_summary.json"
echo "[3/4] Steady-state summary JSON:"
echo "------------------------------------------------------------"
if [ -f "$SUMMARY_JSON" ]; then
    python -m json.tool "$SUMMARY_JSON"
else
    echo "[WARN] Summary JSON not generated: $SUMMARY_JSON"
fi
echo

# Step 4: Compact "TBD-fill" table for the report.
# All key numbers in one place, ready to paste.
echo "[4/4] Report fill-in values:"
echo "------------------------------------------------------------"
python - "$SUMMARY_JSON" "$LOG" "$TS_CSV" << 'PY'
import json, sys, re, csv

summary_path = sys.argv[1]
log_path = sys.argv[2]
ts_path = sys.argv[3]

# Parse summary JSON.
try:
    with open(summary_path) as f:
        s = json.load(f)
except FileNotFoundError:
    print(f"[WARN] Cannot read summary JSON at {summary_path}")
    sys.exit(0)

w = s.get("window") or {}
m = s.get("metrics") or {}
p = s.get("pool_summary") or {}

# Parse simulator log for naive averages.
naive = {}
try:
    with open(log_path) as f:
        for line in f:
            ma = re.search(r"Average generation throughput \(tok/s\):\s+([\d.]+)", line)
            if ma: naive["gen_tps"] = float(ma.group(1))
            ma = re.search(r"Average prompt throughput \(tok/s\):\s+([\d.]+)", line)
            if ma: naive["prompt_tps"] = float(ma.group(1))
            ma = re.search(r"Total token throughput \(tok/s\):\s+([\d.]+)", line)
            if ma: naive["total_tps"] = float(ma.group(1))
            ma = re.search(r"Total latency \(s\):\s+([\d.]+)", line)
            if ma: naive["total_latency_s"] = float(ma.group(1))
            ma = re.search(r"Request throughput \(req/s\):\s+([\d.]+)", line)
            if ma: naive["req_tps"] = float(ma.group(1))
except FileNotFoundError:
    pass

# Steady-state window length as fraction of total sim time.
window_pct = None
if w and naive.get("total_latency_s"):
    window_pct = 100.0 * w.get("duration_s", 0) / naive["total_latency_s"]

print("[Cluster]")
print(f"  Decode pool instances: {p.get('decode_instance_ids')} of {p.get('n_total_instances')} total")
print()

print("[Steady-state window]")
if w:
    print(f"  Start  : {w['t_start_s']:.3f} s")
    print(f"  End    : {w['t_end_s']:.3f} s")
    print(f"  Length : {w['duration_s']:.3f} s"
          + (f"  ({window_pct:.1f}% of total)" if window_pct is not None else ""))
    print(f"  Peak             : {w['peak']}")
    print(f"  Mean concurrent  : {w['mean_concurrent']:.2f}")
    print(f"  Mean waiting     : {w['mean_waiting']:.2f}")
    print(f"  Threshold        : {w['threshold']:.2f}")
else:
    print("  No steady-state window detected.")
print()

print("[Performance metrics — inside steady window]")
if m:
    print(f"  Decode throughput : {m.get('decode_tok_per_s_avg', 0):.1f} tok/s")
    print(f"  Full-in-window    : {m.get('n_full_requests')} (partial: {m.get('n_partial_requests')})")
    tpot_mean = m.get('tpot_mean_ms')
    tpot_p50 = m.get('tpot_p50_ms')
    tpot_p99 = m.get('tpot_p99_ms')
    ttft_mean = m.get('ttft_mean_ms')
    ttft_p99 = m.get('ttft_p99_ms')
    if tpot_mean is not None:
        print(f"  TPOT mean/p50/p99 : {tpot_mean:.2f} / {tpot_p50:.2f} / {tpot_p99:.2f} ms")
    else:
        print("  TPOT              : N/A (window too short — no full-in-window requests)")
    if ttft_mean is not None:
        print(f"  TTFT mean/p99     : {ttft_mean:.2f} / {ttft_p99:.2f} ms")
    else:
        print("  TTFT              : N/A")
else:
    print("  No metrics available (no window detected).")
print()

print("[Naive averages — whole simulation, for comparison]")
if naive:
    print(f"  Total latency     : {naive.get('total_latency_s', 'N/A')} s")
    print(f"  Request rate      : {naive.get('req_tps', 'N/A')} req/s")
    print(f"  Prompt throughput : {naive.get('prompt_tps', 'N/A')} tok/s")
    print(f"  Gen throughput    : {naive.get('gen_tps', 'N/A')} tok/s")
    print(f"  Total throughput  : {naive.get('total_tps', 'N/A')} tok/s")
    # Highlight the gap that justifies the steady-state methodology.
    if m and m.get("decode_tok_per_s_avg") and naive.get("gen_tps"):
        gap_pct = 100.0 * (m["decode_tok_per_s_avg"] - naive["gen_tps"]) / naive["gen_tps"]
        print()
        print(f"  Steady-state gen throughput is {gap_pct:+.1f}% vs naive average")
        print(f"  ({m['decode_tok_per_s_avg']:.1f} vs {naive['gen_tps']:.1f} tok/s)")
else:
    print("  Could not parse simulator log.")
PY

echo
echo "============================================================"
echo "Done. Report-ready files:"
echo "  $SUMMARY_JSON"
echo "  outputs/${RUN_NAME}_decode_pool_ts.png"
echo "  outputs/${RUN_NAME}_decode_pool_ts.csv"
echo "============================================================"