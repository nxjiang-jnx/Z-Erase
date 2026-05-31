"""
ZImage Attention Processors for Knowledge Localization
Adapted from FLUX processors for ZImage's 30-layer single-stream architecture
"""
from enum import Enum, auto
import math
import torch
from typing import Optional
import torch.nn.functional as F

from diffusers.models.attention_dispatch import dispatch_attention_fn

class ZImageCachingAttnProcessor:
    """
    Attention processor for ZImage that caches attention maps and values
    for knowledge localization analysis
    
    ZImage uses single-stream attention with sequence order: [image_tokens, text_tokens]
    - Image tokens: first 1024 positions (fixed)
    - Text tokens: remaining positions (variable length)
    """
    def __init__(self, idx, image_seq_len=1024):
        self.idx = idx
        self.image_seq_len = image_seq_len  # Number of image tokens (usually 1024 for 32x32 latent)
        
        self.attention_maps = []
        self.values = []
        self.out_norms = []
        self.in_norms = []
    
    def clear_maps(self):
        self.attention_maps = []
        self.values = []
        self.out_norms = []
        self.in_norms = []
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        ZImage uses single-stream attention: [image_tokens, text_tokens]
        - Image tokens: positions [0:image_seq_len)
        - Text tokens: positions [image_seq_len:seq_len)
        """
        # Cache input norms for analysis
        with torch.no_grad():
            self.in_norms.append(hidden_states[0].detach().cpu())
        
        # Compute Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        
        # Apply Norms
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
        
        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        
        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        
        # Use dispatch_attention_fn (same as original ZImage)
        from diffusers.models.attention_dispatch import dispatch_attention_fn
        
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=None,
            parallel_config=None,
        )
        
        # Cache attention maps for analysis (compute separately)
        with torch.no_grad():
            scale_factor = 1 / math.sqrt(query.size(-1))
            attention_scores = query @ key.transpose(-2, -1) * scale_factor
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probs = torch.softmax(attention_scores, dim=-1)
            
            # Cache image->text attention
            # Image queries: [:, :, :image_seq_len, :]
            # Text keys/values: [:, :, image_seq_len:, :]
            seq_len = hidden_states.shape[1]
            if seq_len > self.image_seq_len:
                # Extract attention from image queries to text keys
                self.attention_maps.append(
                    attention_probs[:, :, :self.image_seq_len, self.image_seq_len:].detach().cpu()
                )
                # Extract text value vectors
                self.values.append(value[:, self.image_seq_len:, :].detach().cpu())
        
        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        
        # Output projection
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)
        
        # Cache output norms
        with torch.no_grad():
            self.out_norms.append(torch.norm(output[0], dim=1).detach().cpu())
        
        return output


class ZImageAttnContCalculatorProcessor:
    """
    Processor that calculates attention contribution during forward pass
    Used for identifying dominant blocks
    
    ZImage sequence order: [image_tokens (1024), text_tokens (variable)]
    We calculate attention from image queries to target text keys/values
    """
    def __init__(self, token_indices_for_attn_cont_calc, image_seq_len=1024):
        self.image_seq_len = image_seq_len  # Number of image tokens (usually 1024)
        self.token_indices_for_attn_cont_calc = token_indices_for_attn_cont_calc
        
        self.attn_contribution = 0.
        self.attn_contribution_update_count = 0
    
    def calc_attn_cont(self, attention_probs, value, attn):
        """
        Calculate attention contribution for target knowledge tokens
        
        In single-stream, we compute how much image queries attend to target text tokens:
        - Image queries: positions [0:image_seq_len)
        - Text keys/values: positions [image_seq_len:seq_len)
        - Target tokens: specific indices within text tokens (relative to text start)
        """
        assert isinstance(self.token_indices_for_attn_cont_calc, list)
        assert len(self.token_indices_for_attn_cont_calc) > 0
        
        # attention_probs: [batch, heads, seq_len, seq_len]
        # value: [batch, seq_len, heads, head_dim] (after unflatten)
        
        seq_len = attention_probs.shape[2]
        
        # Only calculate if sequence has both image and text tokens
        if seq_len <= self.image_seq_len:
            # No text tokens yet, return zero contribution
            return 0.0
        
        # Extract image->text attention
        # attention_probs[:, :, :image_seq_len, image_seq_len:] gives attention from image queries to text keys
        # m: [batch, heads, image_tokens, text_tokens]
        m = attention_probs[:, :, :self.image_seq_len, self.image_seq_len:].detach().clone()
        
        # Select only the target token columns
        # token_indices are relative to the text token sequence (0-indexed within text)
        m = m[0, :, :, self.token_indices_for_attn_cont_calc]  # [heads, image_tokens, target_tokens]
        
        # Get value vectors for target text tokens
        # value[:, image_seq_len:, :, :] gives text token values
        v = value[:, self.image_seq_len:, :, :].detach().clone()  # [batch, text_tokens, heads, head_dim]
        v = v[0, self.token_indices_for_attn_cont_calc, :, :]  # [target_tokens, heads, head_dim]
        
        # Compute contribution: attention * value
        # m: [heads, image_tokens, target_tokens]
        # v: [target_tokens, heads, head_dim]
        # Transpose v to [heads, target_tokens, head_dim]
        v = v.transpose(0, 1)  # [heads, target_tokens, head_dim]
        
        # Compute: [heads, image_tokens, target_tokens] @ [heads, target_tokens, head_dim]
        # = [heads, image_tokens, head_dim]
        o = torch.einsum('hij,hjk->hik', m, v)  # [heads, image_tokens, head_dim]
        
        # Reshape to [image_tokens, heads * head_dim] for output projection
        o = o.transpose(0, 1).reshape(o.shape[1], o.shape[0] * o.shape[2])
        o = o.to(attn.to_out[0].weight.dtype)
        
        # Apply output projection
        if attn.to_out:
            o = attn.to_out[0](o)
        
        attn_cont = torch.norm(o.to(torch.float32), dim=1).mean().item()
        return attn_cont
    
    def update_attn_cont(self, attn_cont):
        self.attn_contribution += attn_cont
        self.attn_contribution_update_count += 1
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with attention contribution calculation
        
        Calculates attention from image queries to target text keys/values
        """
        # Compute Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        
        # Apply Norms
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
        
        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        
        # Calculate attention contribution WITHOUT mask (for analysis purposes)
        # We calculate raw attention to avoid mask shape issues
        with torch.no_grad():
            scale_factor = 1 / math.sqrt(query.size(-1))
            attention_scores = query @ key.transpose(-2, -1) * scale_factor
            attention_probs_for_contrib = torch.softmax(attention_scores, dim=-1)
            
            # Calculate and update attention contribution
            self.update_attn_cont(self.calc_attn_cont(attention_probs_for_contrib, value, attn))
        
        # Use efficient attention for actual computation WITH mask
        # Prepare mask for dispatch_attention_fn
        attn_mask_for_dispatch = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                attn_mask_for_dispatch = attention_mask[:, None, None, :]
            else:
                attn_mask_for_dispatch = attention_mask
        
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attn_mask_for_dispatch,
            dropout_p=0.0,
            is_causal=False,
            backend=None,
            parallel_config=None,
        )
        
        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        
        # Output projection
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)
        
        return output


class ZImageEmbeddingModifierAttnProcessor:
    """
    Processor that modifies text embeddings during generation
    Used for intervention experiments
    
    ZImage sequence order: [image_tokens (1024), text_tokens (variable)]
    We save/replace the text token portion (positions [image_seq_len:])
    """
    class ProcessorMode(Enum):
        NONE = auto()
        SAVE_TEXT_HIDDEN_STATES = auto()
        REPLACE_TEXT_HIDDEN_STATES = auto()
    
    def __init__(self, mode: ProcessorMode = ProcessorMode.NONE, image_seq_len=1024):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("ZImageEmbeddingModifierAttnProcessor requires PyTorch 2.0")
        
        self.mode = mode
        self.image_seq_len = image_seq_len  # Number of image tokens (usually 1024)
        self.saved_text_hidden_states = None
    
    def clear_cache(self):
        self.saved_text_hidden_states = None
    
    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass with optional text embedding replacement
        
        For single-stream: [image_tokens, text_tokens]
        - Save/Replace text_tokens portion: positions [image_seq_len:]
        - Keep image_tokens portion unchanged: positions [:image_seq_len]
        """
        seq_len = hidden_states.shape[1]
        
        # Save or replace text embeddings based on mode
        if self.mode == self.ProcessorMode.SAVE_TEXT_HIDDEN_STATES:
            # Save text token embeddings (positions after image tokens)
            if seq_len > self.image_seq_len:
                self.saved_text_hidden_states = hidden_states[:, self.image_seq_len:, :].clone()
        elif self.mode == self.ProcessorMode.REPLACE_TEXT_HIDDEN_STATES:
            # Replace text token embeddings with saved clean ones
            if self.saved_text_hidden_states is not None and seq_len > self.image_seq_len:
                hidden_states = hidden_states.clone()
                # Only replace text portion, keep image portion intact
                text_len = min(self.saved_text_hidden_states.shape[1], seq_len - self.image_seq_len)
                hidden_states[:, self.image_seq_len:self.image_seq_len+text_len, :] = self.saved_text_hidden_states[:, :text_len, :]
        
        # Compute Q, K, V
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)
        
        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))
        
        # Apply Norms
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
        
        # Cast to correct dtype
        dtype = query.dtype
        query, key = query.to(dtype), key.to(dtype)
        
        # From [batch, seq_len] to [batch, 1, 1, seq_len] -> broadcast to [batch, heads, seq_len, seq_len]
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        
        # Use dispatch_attention_fn (same as original ZImage)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=None,
            parallel_config=None,
        )
        
        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        
        # Output projection
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:  # dropout
            output = attn.to_out[1](output)
        
        return output
