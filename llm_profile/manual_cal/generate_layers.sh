#!/bin/bash

# ==============================================================================
# Analytical Layer Profiler Script
#
# Generates layers.csv mathematically using Roofline model equations.
# Parameters like --warmup, --repeat, and --device are passed to maintain
# compatibility with the original PyTorch profiler interface but are ignored
# during the analytical computation.
# ==============================================================================

python3 ../profiler/layers/manual_generate_layers.py \
  --hardware WSC-LLM \
  --model "meta-llama/Llama-3.1-8B" \
  --tp-size "4" \
  --max-len 8192 \
  --config-path "custom_hw_model_config.json"