#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_profile
cd llm_profile

#####################  Profile non-attention layers  #####################

CUDA_VISIBLE_DEVICES=0 \
python3 -m profiler.layers.main \
  --hardware T-RTX \
  --model "meta-llama/Llama-3.1-8B" \
  --num-layers 1 \
  --tp-size "1, 4" \
  --warmup 10 \
  --repeat 30 \
  --max-len 6000 \
  --device cuda