#!/bin/bash

# ZImage EAP training script
# Use Position-Masked LoRA for concept erasure

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export HF_HUB_OFFLINE=1

python train_zimage_eap.py \
  --prompt "nude" \
  --start_guidance 3 \
  --negative_guidance 1 \
  --iterations 1000 \
  --lr 1e-4 \
  --devices "4,3" \
  --output_dir "models" \
  --gumbel_lr 1e-3 \
  --gumbel_temp 2 \
  --gumbel_hard 1 \
  --gumbel_k_closest 1000 \
  --gumbel_num_centers 100 \
  --ddim_steps 9 \
  --lora_rank 8
