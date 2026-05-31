# coding: UTF-8
"""
@date: 2025.12.19 - 12.20
@func: ZImage Position-Masked LoRA for concept erasure
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict
from diffusers.models.attention_processor import Attention


# Global storage for text position info (needed during forward)
_text_position_info = {}


def set_text_position_info(image_seq_len: int = 1024):
    """Set global text position info for position-masked LoRA"""
    global _text_position_info
    _text_position_info['image_seq_len'] = image_seq_len


def get_text_position_info():
    """Get global text position info"""
    return _text_position_info.get('image_seq_len', 1024)


class ZImageTextMaskedAttnProcessor:
    """
    Attention processor for ZImage that applies LoRA ONLY on text token positions
    
    Key insight: In ZImage single-stream [Image(1024) | Text(N)]
    - For to_q: Q = W_q @ hidden_states
    - With LoRA: Q = (W_q + LoRA) @ hidden_states
    - We want: Q[:, :1024] = W_q @ image_tokens (no LoRA)
               Q[:, 1024:] = (W_q + LoRA) @ text_tokens (with LoRA)
    """
    
    _attention_backend = None
    _parallel_config = None
    
    def __init__(self, image_seq_len: int = 1024, collect_attn_maps: bool = False):
        self.image_seq_len = image_seq_len
        self.collect_attn_maps = collect_attn_maps
        self.attention_maps = []  # Store for loss calculation
        self._enabled = True  # Control whether to collect maps
    
    def clear_maps(self):
        self.attention_maps = []
    
    def enable_collection(self):
        """Enable attention map collection"""
        self._enabled = True
    
    def disable_collection(self):
        """Disable attention map collection (for neutral prompt forward)"""
        self._enabled = False
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward with position-masked LoRA application
        
        The key trick: LoRA modules in to_q/to_k are already added via TextMaskedLinear
        We don't mask here - masking is done by TextMaskedLinear.forward()
        This processor just handles attention computation and map collection
        """
        # Standard Q/K/V computation (LoRA is applied by TextMaskedLinear automatically)
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        
        # Apply norms
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # Apply RoPE
        def apply_rotary_emb(x_in: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
            with torch.amp.autocast("cuda", enabled=False):
                x = torch.view_as_complex(x_in.float().reshape(*x_in.shape[:-1], -1, 2))
                freqs_cis = freqs_cis.unsqueeze(2)
                x_out = torch.view_as_real(x * freqs_cis).flatten(3)
                return x_out.type_as(x_in)
        
        if freqs_cis is not None:
            query = apply_rotary_emb(query, freqs_cis)
            key = apply_rotary_emb(key, freqs_cis)
        
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        
        # Collect attention maps if enabled AND collection is turned on
        # CRITICAL: Only collect when _enabled is True (during concept forward only)
        if self.collect_attn_maps and self._enabled:
            batch_size, seq_len, n_heads, head_dim = query.shape
            if seq_len > self.image_seq_len:
                scale_factor = 1 / math.sqrt(query.size(-1))
                query_for_attn = query.transpose(1, 2)
                key_for_attn = key.transpose(1, 2)
                attention_scores = query_for_attn @ key_for_attn.transpose(-2, -1) * scale_factor
                
                if attention_mask is not None:
                    if attention_mask.ndim == 2:
                        attention_mask_expanded = attention_mask[:, None, None, :]
                    else:
                        attention_mask_expanded = attention_mask
                    attention_scores = attention_scores + attention_mask_expanded
                
                attention_probs = torch.softmax(attention_scores, dim=-1)
                
                # Extract image->text attention
                # Shape: [batch, heads, image_tokens (1024), text_tokens]
                image_to_text_attn = attention_probs[:, :, :self.image_seq_len, self.image_seq_len:]
                self.attention_maps.append(image_to_text_attn)
        
        # Prepare mask for efficient attention
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        
        # Use efficient attention
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        
        return output


class TextMaskedLinear(nn.Module):
    """
    Wrapper for Linear layer that applies LoRA ONLY on text positions
    
    Forward: 
    - For positions [:image_seq_len]: use base_weight only
    - For positions [image_seq_len:]: use base_weight + lora_delta
    """
    
    def __init__(self, base_linear: nn.Linear, image_seq_len: int = 1024):
        super().__init__()
        self.base_linear = base_linear
        self.image_seq_len = image_seq_len
        
        # LoRA parameters (initialized to zero, so no initial change)
        self.lora_down = nn.Linear(base_linear.in_features, 64, bias=False)
        self.lora_up = nn.Linear(64, base_linear.out_features, bias=False)
        self.lora_scale = 1.0
        
        # Initialize LoRA to zero effect
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        
        # Flag to disable LoRA (for computing original model output)
        self._lora_enabled = True
    
    def enable_lora(self):
        """Enable LoRA (default)"""
        self._lora_enabled = True
    
    def disable_lora(self):
        """Disable LoRA to get original model output"""
        self._lora_enabled = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [batch, seq_len, hidden_dim]
        
        Returns:
        - For image positions: W @ x_img
        - For text positions: (W + LoRA) @ x_text  (if LoRA enabled)
        
        Note: LoRA parameters are kept in float32 for training stability,
        but are automatically cast to input dtype during forward pass.
        """
        batch_size, seq_len, hidden_dim = x.shape
        input_dtype = x.dtype
        
        # Base output for all positions
        base_output = self.base_linear(x)
        
        # If LoRA disabled or only image tokens, return base output
        if not self._lora_enabled or seq_len <= self.image_seq_len:
            return base_output
        
        # Apply LoRA ONLY to text positions
        text_x = x[:, self.image_seq_len:, :]
        
        # Compute LoRA delta with proper dtype handling
        # LoRA params may be float32 (for training), but we cast to input dtype for computation
        if self.lora_down.weight.dtype != input_dtype:
            lora_down_weight = self.lora_down.weight.to(dtype=input_dtype)
            lora_up_weight = self.lora_up.weight.to(dtype=input_dtype)
            lora_delta = torch.nn.functional.linear(
                torch.nn.functional.linear(text_x, lora_down_weight, None),
                lora_up_weight,
                None
            ) * self.lora_scale
        else:
            lora_delta = self.lora_up(self.lora_down(text_x)) * self.lora_scale
        
        # Output: image positions unchanged, text positions get LoRA
        output = base_output.clone()
        output[:, self.image_seq_len:, :] = output[:, self.image_seq_len:, :] + lora_delta
        
        return output


def apply_text_masked_lora_to_transformer(
    transformer,
    target_layers: List[int] = None,
    lora_rank: int = 64,
    image_seq_len: int = 1024,
):
    """
    Apply position-masked LoRA to ZImage transformer
    
    Args:
        transformer: ZImageTransformer2DModel
        target_layers: Which unified layers to apply LoRA (default: all)
        lora_rank: LoRA rank
        image_seq_len: Number of image tokens (fixed at 1024 for ZImage)
    
    Returns:
        List of trainable parameters
    """
    set_text_position_info(image_seq_len)
    
    if target_layers is None:
        target_layers = list(range(len(transformer.layers)))
    
    # Get device from transformer
    device = next(transformer.parameters()).device
    
    trainable_params = []
    
    for idx in target_layers:
        layer = transformer.layers[idx]
        attn = layer.attention
        
        # Replace to_q with TextMaskedLinear
        original_to_q = attn.to_q
        masked_to_q = TextMaskedLinear(original_to_q, image_seq_len)
        masked_to_q.lora_down = nn.Linear(original_to_q.in_features, lora_rank, bias=False)
        masked_to_q.lora_up = nn.Linear(lora_rank, original_to_q.out_features, bias=False)
        nn.init.kaiming_uniform_(masked_to_q.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(masked_to_q.lora_up.weight)
        # Keep LoRA params in float32 for training stability, but move to device
        masked_to_q.lora_down = masked_to_q.lora_down.to(device=device, dtype=torch.float32)
        masked_to_q.lora_up = masked_to_q.lora_up.to(device=device, dtype=torch.float32)
        masked_to_q.base_linear = masked_to_q.base_linear.to(device=device)
        masked_to_q = masked_to_q.to(device=device)
        attn.to_q = masked_to_q
        
        # Replace to_k with TextMaskedLinear
        original_to_k = attn.to_k
        masked_to_k = TextMaskedLinear(original_to_k, image_seq_len)
        masked_to_k.lora_down = nn.Linear(original_to_k.in_features, lora_rank, bias=False)
        masked_to_k.lora_up = nn.Linear(lora_rank, original_to_k.out_features, bias=False)
        nn.init.kaiming_uniform_(masked_to_k.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(masked_to_k.lora_up.weight)
        masked_to_k.lora_down = masked_to_k.lora_down.to(device=device, dtype=torch.float32)
        masked_to_k.lora_up = masked_to_k.lora_up.to(device=device, dtype=torch.float32)
        masked_to_k.base_linear = masked_to_k.base_linear.to(device=device)
        masked_to_k = masked_to_k.to(device=device)
        attn.to_k = masked_to_k
        
        # Collect trainable params
        trainable_params.extend([
            masked_to_q.lora_down.weight,
            masked_to_q.lora_up.weight,
            masked_to_k.lora_down.weight,
            masked_to_k.lora_up.weight,
        ])
        
        # Set processor with attention map collection
        processor = ZImageTextMaskedAttnProcessor(
            image_seq_len=image_seq_len,
            collect_attn_maps=True
        )
        attn.processor = processor
    
    # Freeze base model
    for param in transformer.parameters():
        param.requires_grad = False
    
    # Enable gradients for LoRA
    for param in trainable_params:
        param.requires_grad = True
    
    return trainable_params


def enable_attention_collection(transformer):
    """Enable attention map collection for all layers"""
    for layer in transformer.layers:
        processor = layer.attention.processor
        if isinstance(processor, ZImageTextMaskedAttnProcessor):
            processor.enable_collection()


def disable_attention_collection(transformer):
    """Disable attention map collection for all layers (use for neutral forward)"""
    for layer in transformer.layers:
        processor = layer.attention.processor
        if isinstance(processor, ZImageTextMaskedAttnProcessor):
            processor.disable_collection()


def enable_lora(transformer):
    """Enable LoRA in all layers"""
    for layer in transformer.layers:
        attn = layer.attention
        if hasattr(attn.to_q, 'enable_lora'):
            attn.to_q.enable_lora()
        if hasattr(attn.to_k, 'enable_lora'):
            attn.to_k.enable_lora()


def disable_lora(transformer):
    """Disable LoRA in all layers (to get original model output)"""
    for layer in transformer.layers:
        attn = layer.attention
        if hasattr(attn.to_q, 'disable_lora'):
            attn.to_q.disable_lora()
        if hasattr(attn.to_k, 'disable_lora'):
            attn.to_k.disable_lora()


def collect_attention_maps(transformer) -> torch.Tensor:
    """
    Collect attention maps from all layers
    
    Returns: [num_layers, batch, heads, image_tokens, text_tokens]
    """
    all_maps = []
    for layer in transformer.layers:
        processor = layer.attention.processor
        if isinstance(processor, ZImageTextMaskedAttnProcessor):
            if len(processor.attention_maps) > 0:
                # Concatenate maps from this layer
                layer_maps = torch.cat(processor.attention_maps, dim=0)
                all_maps.append(layer_maps)
    
    if len(all_maps) == 0:
        return None
    
    return torch.stack(all_maps, dim=0)


def clear_attention_maps(transformer):
    """Clear stored attention maps from all layers"""
    for layer in transformer.layers:
        processor = layer.attention.processor
        if isinstance(processor, ZImageTextMaskedAttnProcessor):
            processor.clear_maps()


def calculate_text_masked_attn_loss(
    attention_maps: torch.Tensor,
    remove_indices: List[int],
) -> torch.Tensor:
    """
    Calculate attention loss for target token positions
    
    This is equivalent to FLUX's attention loss but for ZImage
    
    Args:
        attention_maps: [num_layers, batch, heads, image_tokens, text_tokens]
        remove_indices: Token indices in text sequence to suppress
    
    Returns:
        Loss value to minimize attention to target tokens
    """
    if attention_maps is None or len(remove_indices) == 0:
        return torch.tensor(0.0)
    
    # Extract attention to target tokens
    # attention_maps[..., remove_indices] -> [..., len(remove_indices)]
    target_attn = attention_maps[:, :, :, :, remove_indices]
    
    # L2 norm to minimize attention to these positions
    loss = torch.norm(target_attn, p=2)
    
    return loss


def get_lora_state_dict(transformer) -> Dict[str, torch.Tensor]:
    """Get LoRA weights from transformer for saving"""
    lora_state_dict = {}
    for idx, layer in enumerate(transformer.layers):
        attn = layer.attention
        if hasattr(attn.to_q, 'lora_down'):
            lora_state_dict[f'layers.{idx}.attention.to_q.lora_down.weight'] = attn.to_q.lora_down.weight.data.clone()
            lora_state_dict[f'layers.{idx}.attention.to_q.lora_up.weight'] = attn.to_q.lora_up.weight.data.clone()
        if hasattr(attn.to_k, 'lora_down'):
            lora_state_dict[f'layers.{idx}.attention.to_k.lora_down.weight'] = attn.to_k.lora_down.weight.data.clone()
            lora_state_dict[f'layers.{idx}.attention.to_k.lora_up.weight'] = attn.to_k.lora_up.weight.data.clone()
    return lora_state_dict


def get_lora_param_stats(transformer) -> Dict[str, float]:
    """Get statistics of LoRA parameters for debugging"""
    stats = {}
    total_norm = 0.0
    total_grad_norm = 0.0
    
    for idx, layer in enumerate(transformer.layers):
        attn = layer.attention
        if hasattr(attn.to_q, 'lora_down'):
            q_down_norm = attn.to_q.lora_down.weight.data.norm().item()
            q_up_norm = attn.to_q.lora_up.weight.data.norm().item()
            total_norm += q_down_norm + q_up_norm
            
            if attn.to_q.lora_down.weight.grad is not None:
                total_grad_norm += attn.to_q.lora_down.weight.grad.norm().item()
            if attn.to_q.lora_up.weight.grad is not None:
                total_grad_norm += attn.to_q.lora_up.weight.grad.norm().item()
        
        if hasattr(attn.to_k, 'lora_down'):
            k_down_norm = attn.to_k.lora_down.weight.data.norm().item()
            k_up_norm = attn.to_k.lora_up.weight.data.norm().item()
            total_norm += k_down_norm + k_up_norm
            
            if attn.to_k.lora_down.weight.grad is not None:
                total_grad_norm += attn.to_k.lora_down.weight.grad.norm().item()
            if attn.to_k.lora_up.weight.grad is not None:
                total_grad_norm += attn.to_k.lora_up.weight.grad.norm().item()
    
    stats['param_norm'] = total_norm
    stats['grad_norm'] = total_grad_norm
    
    return stats
