#!/usr/bin/env bash
# run_load_scale.sh — load scaling sweep (Track B baseline setup)
#
# Usage (from repo root, servingsim conda env):
#   conda activate servingsim
#   bash experiments/run_load_scale.sh
#
# Scaling factors applied to inter-arrival time intervals:
#   scale < 1.0  →  shorter gaps  →  higher load
#   scale > 1.0  →  longer gaps   →  lower load
#
# Baseline: azure_trace_conv_llama (Track B simA setup)
# Peak concurrent decode at baseline = 10 (max_num_seqs=10)

set -euo pipefail

CLUSTER_CONFIG="configs/cluster/custom_disaggregated.json"
DATASET="workloads/azure_trace_conv_llama.jsonl"
NUM_REQS=300
MAX_NUM_SEQS=10          # Required: 32 GB NPU OOMs at default 128
OUTDIR="outputs/load_scale_exp"
LOG_INTERVAL=1.0

mkdir -p "$OUTDIR"

# Scaling factors: 0.25 (4x load), 0.5 (2x), 1.0 (baseline), 2.0 (0.5x), 4.0 (0.25x)
SCALES=(0.25 0.5 1.0 2.0 4.0)

for SCALE in "${SCALES[@]}"; do
    TAG="scale${SCALE}"
    OUT_CSV="${OUTDIR}/${TAG}.csv"
    LOG_FILE="${OUTDIR}/${TAG}.log"

    echo "=== Running load_scale=${SCALE} (tag: ${TAG}) ==="
    python -m serving \
        --cluster-config "$CLUSTER_CONFIG" \
        --dataset "$DATASET" \
        --num-reqs "$NUM_REQS" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --output "$OUT_CSV" \
        --load-scale "$SCALE" \
        --log-interval "$LOG_INTERVAL" \
        --log-level WARNING \
        2>&1 | tee "$LOG_FILE"

    echo "  -> CSV: $OUT_CSV"
    echo ""
done

echo "All runs complete. Results in: $OUTDIR"
echo "Run: python experiments/plot_load_scale.py to visualize."
