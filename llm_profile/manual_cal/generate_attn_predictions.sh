#!/bin/bash

# ==============================================================================
# Analytical Attention Profiler Script
#
# Generates attn_prefill_predictions.csv and attn_decode_predictions.csv
# using the exact conditions provided.
# ==============================================================================

python3 ../profiler/attention/manual_generate_attn_predictions.py \
  --model "meta-llama/Llama-3.1-8B" \
  --hardware WSC-LLM \
  --max-len 2048 \
  --tp-size "4" \
  --config-path "custom_hw_model_config.json"