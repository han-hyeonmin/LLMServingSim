#!/bin/bash

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

# Run custom disaggregated serving simulation
python main.py --cluster-config 'cluster_config/custom_disaggregated.json' \
    --fp 16 --block-size 16 \
    --dataset 'dataset/fixed_in128_out512_req256_rate10.jsonl' \
    --output 'output/custom_pdd_run.csv' \
    --num-req 1 \
    --log-interval 1.0