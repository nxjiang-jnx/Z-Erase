#!/bin/bash

# UCE for ZImage - Erase Nudity
# Example: Remove Nudity

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=3

python train_uce_lora.py \
    --model_id "Tongyi-MAI/Z-Image-Turbo" \
    --concept_type "nudity" \
    --edit_concepts "nude;naked" \
    --guide_concepts "person;person" \
    --preserve_concepts "person;hair" \
    --erase_scale 10.0 \
    --preserve_scale 1.0 \
    --lamb 0.1 \
    --lora_rank 64 \
    --target_layers "19, 22, 21, 20, 17, 16" \
    --expand_prompts "true" \
    --save_dir "uce_models" \
    --exp_name "text_masked_lora_nudity" \
    --device "cuda:0"

echo ""
echo "✓ Nudity erasure completed!"

