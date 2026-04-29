#!/bin/bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llm_profile
cd llm_profile

#######################  Profile attention layers  #######################

CUDA_VISIBLE_DEVICES=0 \
python -m profiler.attention.main \
  --model "meta-llama/Llama-3.1-8B" \
  --hardware T-RTX \
  --tp-size "4" \
  \
  --max-len 8192 \
  --min-batch-size 1 \
  --max-batch-size 64 \
  \
  --warmup 20 \
  --repeat 50 \
  --profile-method cuda_event \
  --device cuda