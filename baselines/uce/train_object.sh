#!/bin/bash

# UCE for ZImage - Erase Object
# Example: Remove specific objects

export HF_ENDPOINT="https://hf-mirror.com"
export CUDA_VISIBLE_DEVICES=4

python train_uce_lora.py \
    --model_id "Tongyi-MAI/Z-Image-Turbo" \
    --concept_type "object" \
    --edit_concepts "blood;gun;weapon;rifle;violence" \
    --guide_concepts "object" \
    --preserve_concepts "person;hand;toy" \
    --erase_scale 10.0 \
    --preserve_scale 1.0 \
    --lamb 0.1 \
    --lora_rank 64 \
    --target_layers "19, 22, 21, 20, 17, 16" \
    --expand_prompts "false" \
    --save_dir "uce_models" \
    --exp_name "text_masked_lora_violence" \
    --device "cuda:0"

echo ""
echo "✓ Object erasure completed!"

