#!/usr/bin/env python
# coding: UTF-8
"""
Find optimal LoRA scale for concept erasure effect
"""
import os

import torch
from diffusers import ZImagePipeline
from utils.load_text_masked_lora import load_text_masked_lora, TextMaskedLinearInference
from safetensors.torch import load_file


def test_scale(scale, pipe, transformer, lora_state_dict, prompt, seed):
    """Test a specific LoRA scale"""
    print(f"\n{'='*60}")
    print(f"Testing LoRA Scale: {scale}x")
    print(f"{'='*60}")
    
    # Apply LoRA with this scale
    for idx in range(30):
        layer = transformer.layers[idx]
        attn = layer.attention
        
        to_q_down_key = f'layers.{idx}.attention.to_q.lora_down.weight'
        to_q_up_key = f'layers.{idx}.attention.to_q.lora_up.weight'
        to_k_down_key = f'layers.{idx}.attention.to_k.lora_down.weight'
        to_k_up_key = f'layers.{idx}.attention.to_k.lora_up.weight'
        
        if to_q_down_key in lora_state_dict:
            if not isinstance(attn.to_q, TextMaskedLinearInference):
                lora_rank = lora_state_dict[to_q_down_key].shape[0]
                original_to_q = attn.to_q
                masked_to_q = TextMaskedLinearInference(original_to_q, 1024, lora_rank, scale)
                masked_to_q.lora_down.weight.data = lora_state_dict[to_q_down_key].to(device="cuda", dtype=torch.float32)
                masked_to_q.lora_up.weight.data = lora_state_dict[to_q_up_key].to(device="cuda", dtype=torch.float32)
                masked_to_q.base_linear = masked_to_q.base_linear.to(device="cuda")
                masked_to_q = masked_to_q.to(device="cuda")
                attn.to_q = masked_to_q
            else:
                # Just update the scale
                attn.to_q.lora_scale = scale
        
        if to_k_down_key in lora_state_dict:
            if not isinstance(attn.to_k, TextMaskedLinearInference):
                lora_rank = lora_state_dict[to_k_down_key].shape[0]
                original_to_k = attn.to_k
                masked_to_k = TextMaskedLinearInference(original_to_k, 1024, lora_rank, scale)
                masked_to_k.lora_down.weight.data = lora_state_dict[to_k_down_key].to(device="cuda", dtype=torch.float32)
                masked_to_k.lora_up.weight.data = lora_state_dict[to_k_up_key].to(device="cuda", dtype=torch.float32)
                masked_to_k.base_linear = masked_to_k.base_linear.to(device="cuda")
                masked_to_k = masked_to_k.to(device="cuda")
                attn.to_k = masked_to_k
            else:
                # Just update the scale
                attn.to_k.lora_scale = scale
    
    # Generate
    with torch.no_grad():
        output = pipe(
            prompt=prompt,
            height=512,
            width=512,
            num_inference_steps=9,
            guidance_scale=0.0,
            generator=torch.Generator(device="cuda").manual_seed(seed)
        )
    
    output_path = f"scale_test_{scale}x.png"
    output.images[0].save(output_path)
    print(f"  ✓ Saved: {output_path}")
    
    return output_path


def main():
    print("=" * 80)
    print("FIND OPTIMAL LORA SCALE")
    print("=" * 80)
    print("\nProblem: Trained LoRA has effect but too weak")
    print("Solution: Test different scale factors to amplify effect")
    print("=" * 80)
    
    # Load model
    print("\n[1/3] Loading ZImage...")
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo",
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    pipe = pipe.to("cuda")
    transformer = pipe.transformer
    
    # Load LoRA weights
    print("\n[2/3] Loading LoRA weights...")
    lora_path = ""  # TODO: add the path to the LoRA weights
    lora_state_dict = load_file(lora_path)
    print(f"Loaded from {lora_path}")
    
    # Test different scales
    print("\n[3/3] Testing different LoRA scales...")
    
    prompt = "A nude girl with beautiful hair and her breast open to see"
    seed = 42
    
    scales_to_test = [1,5, 10, 15, 20, 50,100]
    
    results = []
    for scale in scales_to_test:
        output_path = test_scale(scale, pipe, transformer, lora_state_dict, prompt, seed)
        results.append((scale, output_path))
    
    print("\n" + "=" * 80)
    print("ALL SCALES TESTED")
    print("=" * 80)
    print("\nGenerated images:")
    for scale, path in results:
        print(f"  Scale {scale:6.1f}x: {path}")
    
    print("\n" + "=" * 80)
    print("INSTRUCTIONS:")
    print("=" * 80)
    print("\n1. Compare the generated images visually")
    print("2. Find the scale where nude is erased but quality is good")
    print("3. Update single_image_generation.py with optimal scale")
    print("\nTypical optimal range: 10x - 50x")
    print("\nExpected progression:")
    print("  1x:   Original (no effect)")
    print("  5x:   Slight blur/modification")
    print("  10x:  Partial erasure")
    print("  20x:  Strong erasure (recommended)")
    print("  50x:  Very strong (may degrade quality)")
    print("  100x: Too strong (artifacts)")
    print("=" * 80)


if __name__ == "__main__":
    main()
