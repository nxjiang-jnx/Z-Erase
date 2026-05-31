#!/bin/bash

# UCE for ZImage - Erase Artist Style
# Example: Remove Van Gogh style

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=6

python train_uce_lora.py \
    --model_id "Tongyi-MAI/Z-Image-Turbo" \
    --concept_type "art" \
    --edit_concepts "Van Gogh" \
    --guide_concepts "art" \
    --preserve_concepts "painting;landscape;portrait" \
    --erase_scale 10.0 \
    --preserve_scale 1.0 \
    --lamb 0.1 \
    --lora_rank 64 \
    --target_layers "19, 22, 21, 20, 17, 16" \
    --expand_prompts "true" \
    --save_dir "uce_models" \
    --exp_name "text_masked_lora_vangogh" \
    --device "cuda:0"

echo ""
echo "✓ Artist style erasure completed!"

