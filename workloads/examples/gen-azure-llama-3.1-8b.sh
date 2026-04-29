#!/bin/bash
# Generate azure_trace_conv_llama.jsonl from the Azure LLM Inference Public Trace.
#
# Place the raw CSV files in workloads/azurepublicdataset/ before running:
#   AzureLLMInferenceTrace_conv.csv
#   AzureLLMInferenceTrace_code.csv
#
# HF_TOKEN must be set in the environment (Llama-3.1-8B is a gated model).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DATASET="${DATASET:-conv}"

python3 -m workloads.generators azure \
    --dataset "$DATASET" \
    --model "$MODEL" \
    --max-input-toks 2048 \
    --max-kv-toks 4096 \
    --output workloads/azure_trace_${DATASET}_llama.jsonl

    # Uncomment to cap request count:
    # --max-requests 100 \

    # Uncomment for batch / offline mode (all arrival_time_ns = 0):
    # --all-arrives-at-0 \