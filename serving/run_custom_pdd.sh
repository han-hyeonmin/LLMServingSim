#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate servingsim

# ==============================================================================
# Custom Prefill-Decode Disaggregated (PDD) Serving Simulation
#
# Topology:
#   - 1 Node
#   - 2 Prefill Instances (TP=1, TP=2)
#   - 1 Decode Instance (TP=4)
#   - KV Cache Transfer: 112 GB/s Link bandwidth
#
# Purpose:
#   Evaluate the throughput and latency of an asymmetric disaggregated
#   serving architecture with different TP scales per pool.
# ==============================================================================

python -m serving --cluster-config 'configs/cluster/custom_disaggregated.json' \
    --dtype float16 --block-size 16 \
    --dataset 'workloads/azure_trace_conv_llama.jsonl' \
    --output 'outputs/custom_pdd_run_all_arrives_at_0.csv' \
    --num-req 100