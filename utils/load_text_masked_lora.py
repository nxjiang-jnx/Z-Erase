# coding: UTF-8
"""
@date: 2025.12.20
@func: Load and apply trained Position-Masked LoRA weights for ZImage inference
"""

import torch
import torch.nn as nn
import math
from safetensors.torch import load_file
from typing import Dict, Optional
from diffusers import ZImagePipeline


class TextMaskedLinearInference(nn.Module):
    
    def __init__(self, base_linear: nn.Linear, image_seq_len: int = 1024, lora_rank: int = 64, lora_scale: float = 1.0):
        super().__init__()
        self.base_linear = base_linear
        self.image_seq_len = image_seq_len
        
        # LoRA parameters (will be loaded from checkpoint)
        self.lora_down = nn.Linear(base_linear.in_features, lora_rank, bias=False)
        self.lora_up = nn.Linear(lora_rank, base_linear.out_features, bias=False)
        self.lora_scale = lora_scale  # Configurable scale factor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, hidden_dim]
        
        Returns:
        - For image positions: W @ x_img
        - For text positions: (W + LoRA) @ x_text
        
        Note: LoRA parameters are kept in float32 (from training),
        but are automatically cast to input dtype during forward pass.
        """
        batch_size, seq_len, hidden_dim = x.shape
        input_dtype = x.dtype
        
        # Base output for all positions
        base_output = self.base_linear(x)
        
        if seq_len <= self.image_seq_len:
            # Only image tokens, no LoRA
            return base_output
        
        # Apply LoRA ONLY to text positions
        text_x = x[:, self.image_seq_len:, :]
        
        # Compute LoRA delta with proper dtype handling
        # LoRA params may be float32 (from training), but we cast to input dtype for computation
        if self.lora_down.weight.dtype != input_dtype:
            # Cast LoRA weights to input dtype for this forward pass
            lora_down_weight = self.lora_down.weight.to(dtype=input_dtype)
            lora_up_weight = self.lora_up.weight.to(dtype=input_dtype)
            lora_delta = torch.nn.functional.linear(
                torch.nn.functional.linear(text_x, lora_down_weight, None),
                lora_up_weight,
                None
            ) * self.lora_scale
        else:
            # Same dtype, use normal forward
            lora_delta = self.lora_up(self.lora_down(text_x)) * self.lora_scale
        
        # Output: image positions unchanged, text positions get LoRA
        output = base_output.clone()
        output[:, self.image_seq_len:, :] = output[:, self.image_seq_len:, :] + lora_delta
        
        return output


def load_text_masked_lora(
    pipe: ZImagePipeline,
    lora_path: str,
    image_seq_len: int = 1024,
    device: str = "cuda",
    lora_scale: float = 10.0,  # Increased default scale for stronger effect
) -> ZImagePipeline:
    """
    Load trained position-masked LoRA weights into ZImage pipeline
    """
    # Load LoRA weights
    lora_state_dict = load_file(lora_path)
    
    transformer = pipe.transformer
    
    # Parse which layers have LoRA
    layers_with_lora = set()
    for key in lora_state_dict.keys():
        # Format: layers.{idx}.attention.to_q.lora_down.weight
        parts = key.split('.')
        if len(parts) >= 2 and parts[0] == 'layers':
            layers_with_lora.add(int(parts[1]))
    
    print(f"[Load LoRA] Found LoRA weights for layers: {sorted(layers_with_lora)}")
    
    # Apply LoRA to each layer
    for idx in layers_with_lora:
        layer = transformer.layers[idx]
        attn = layer.attention
        
        # Get LoRA rank from weights
        to_q_down_key = f'layers.{idx}.attention.to_q.lora_down.weight'
        to_q_up_key = f'layers.{idx}.attention.to_q.lora_up.weight'
        to_k_down_key = f'layers.{idx}.attention.to_k.lora_down.weight'
        to_k_up_key = f'layers.{idx}.attention.to_k.lora_up.weight'
        
        if to_q_down_key in lora_state_dict:
            lora_rank = lora_state_dict[to_q_down_key].shape[0]
            
            # Create and apply TextMaskedLinear for to_q with configurable scale
            original_to_q = attn.to_q
            masked_to_q = TextMaskedLinearInference(original_to_q, image_seq_len, lora_rank, lora_scale)
            # Load LoRA weights as float32 (matching training dtype)
            # The forward pass will handle dtype conversion automatically
            masked_to_q.lora_down.weight.data = lora_state_dict[to_q_down_key].to(device=device, dtype=torch.float32)
            masked_to_q.lora_up.weight.data = lora_state_dict[to_q_up_key].to(device=device, dtype=torch.float32)
            # Move base_linear to device (it will use model's dtype)
            masked_to_q.base_linear = masked_to_q.base_linear.to(device=device)
            masked_to_q = masked_to_q.to(device=device)
            attn.to_q = masked_to_q
        
        if to_k_down_key in lora_state_dict:
            lora_rank = lora_state_dict[to_k_down_key].shape[0]
            
            # Create and apply TextMaskedLinear for to_k with configurable scale
            original_to_k = attn.to_k
            masked_to_k = TextMaskedLinearInference(original_to_k, image_seq_len, lora_rank, lora_scale)
            # Load LoRA weights as float32 (matching training dtype)
            masked_to_k.lora_down.weight.data = lora_state_dict[to_k_down_key].to(device=device, dtype=torch.float32)
            masked_to_k.lora_up.weight.data = lora_state_dict[to_k_up_key].to(device=device, dtype=torch.float32)
            # Move base_linear to device (it will use model's dtype)
            masked_to_k.base_linear = masked_to_k.base_linear.to(device=device)
            masked_to_k = masked_to_k.to(device=device)
            attn.to_k = masked_to_k
    
    print(f"[Load LoRA] Successfully applied position-masked LoRA to {len(layers_with_lora)} layers")
    
    return pipe


def generate_with_erased_concept(
    pipe: ZImagePipeline,
    prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 9,
    guidance_scale: float = 0.0,
    seed: Optional[int] = None,
):
    """
    Generate image with erased concept using position-masked LoRA
    """
    generator = None
    if seed is not None:
        generator = torch.Generator(device=pipe.device).manual_seed(seed)
    
    output = pipe(
        prompt=prompt,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    
    return output.images[0]


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True, help="Path to ZImage model")
    parser.add_argument("--lora_path", type=str, required=True, help="Path to text_masked_lora.safetensors")
    parser.add_argument("--prompt", type=str, default="a beautiful woman", help="Prompt to generate")
    parser.add_argument("--output", type=str, default="output.png", help="Output image path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Load pipeline
    print(f"Loading ZImage from {args.model_path}...")
    pipe = ZImagePipeline.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
    ).to("cuda")
    
    # Load LoRA
    print(f"Loading LoRA from {args.lora_path}...")
    pipe = load_text_masked_lora(pipe, args.lora_path)
    
    # Generate
    print(f"Generating: {args.prompt}")
    image = generate_with_erased_concept(
        pipe,
        args.prompt,
        seed=args.seed,
    )
    
    image.save(args.output)
    print(f"Saved to {args.output}")

