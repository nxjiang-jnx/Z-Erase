# coding: UTF-8
"""
    @date: 2025.12.5-12.14
    @func: Loss calculation for ZImage concept erasure
"""

import random
import math
import torch
import torch.nn.functional as F
from typing import List, Optional
from enum import Enum, auto
from diffusers import AutoencoderKL, ZImagePipeline
from diffusers.models.attention_processor import Attention

from .esd_utils import latent_sample, predict_noise, zimage_pack_latents, _prepare_latent_image_ids


class ZImageEmbeddingReplacementProcessor:
    """
    Attention processor for ZImage that replaces text embeddings in specified layers
    Used for LIAL (Latent Injection Alignment Loss)
    
    ZImage sequence: [image_tokens (1024), text_tokens (variable)]
    We replace text token embeddings while keeping image tokens intact
    """
    class ProcessorMode(Enum):
        NONE = auto()
        SAVE_TEXT_HIDDEN_STATES = auto()
        REPLACE_TEXT_HIDDEN_STATES = auto()
    
    def __init__(self, mode: ProcessorMode = ProcessorMode.NONE, image_seq_len: int = 1024):
        self.mode = mode
        self.image_seq_len = image_seq_len
        self.saved_text_hidden_states = None
        self._attention_backend = None
        self._parallel_config = None
    
    def clear_cache(self):
        """Clear saved text hidden states"""
        self.saved_text_hidden_states = None
    
    def __call__(
        self,
        attn: Attention,
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
            # Replace text token embeddings with saved anchor ones
            if self.saved_text_hidden_states is not None and seq_len > self.image_seq_len:
                hidden_states = hidden_states.clone()
                # Only replace text portion, keep image portion intact
                text_len = min(self.saved_text_hidden_states.shape[1], seq_len - self.image_seq_len)
                hidden_states[:, self.image_seq_len:self.image_seq_len+text_len, :] = self.saved_text_hidden_states[:, :text_len, :]
        
        # Standard forward computation
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
        
        # Prepare attention mask for dispatch
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
        
        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        
        # Output projection
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        
        return output


class ZImageAttnMapCollectorProcessor:
    
    def __init__(self, image_seq_len: int = 1024):
        self.image_seq_len = image_seq_len
        self._attention_backend = None
        self._parallel_config = None
        
        # Storage for attention maps
        self.attention_maps = []  # List of [batch, heads, image_tokens, text_tokens]
    
    def clear_maps(self):
        """Clear stored attention maps"""
        self.attention_maps = []
    
    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        freqs_cis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with attention map collection"""
        # Standard forward computation
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
        
        # === COLLECT ATTENTION MAPS ===
        # Compute attention scores manually to collect maps
        batch_size, seq_len, n_heads, head_dim = query.shape
        
        if seq_len > self.image_seq_len:
            # Only collect when both image and text tokens are present
            scale_factor = 1 / math.sqrt(query.size(-1))
            
            # Compute attention scores: [batch, seq_len, heads, head_dim] -> [batch, heads, seq_len, seq_len]
            query_for_attn = query.transpose(1, 2)  # [batch, heads, seq_len, head_dim]
            key_for_attn = key.transpose(1, 2)      # [batch, heads, seq_len, head_dim]
            
            attention_scores = query_for_attn @ key_for_attn.transpose(-2, -1) * scale_factor
            
            # Apply mask if present
            if attention_mask is not None:
                if attention_mask.ndim == 2:
                    attention_mask_expanded = attention_mask[:, None, None, :]
                else:
                    attention_mask_expanded = attention_mask
                attention_scores = attention_scores + attention_mask_expanded
            
            attention_probs = torch.softmax(attention_scores, dim=-1)
            
            # Extract image->text attention: [batch, heads, image_tokens, text_tokens]
            # Image queries: [:, :, :image_seq_len, :]
            # Text keys: [:, :, :, image_seq_len:]
            image_to_text_attn = attention_probs[:, :, :self.image_seq_len, self.image_seq_len:]
            
            # Store for loss calculation
            self.attention_maps.append(image_to_text_attn)
        
        # Prepare attention mask for dispatch_attention_fn
        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = attention_mask[:, None, None, :]
        
        # Use efficient attention for actual forward pass
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
        
        # Reshape back
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(dtype)
        
        # Output projection
        output = attn.to_out[0](hidden_states)
        if len(attn.to_out) > 1:
            output = attn.to_out[1](output)
        
        return output


def calculate_esd_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, 
                       transformer, noise_scheduler, prompts, vae, criteria, 
                       negative_guidance, weight_dtype, neg_prompts, start_guidance=0, 
                       ddim_steps=9, latents_cache=None, step=0):
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents and latents_cache is not None:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    # Get conditional embeddings for the prompt
    # e_0: embedding for negative/unconditional prompt
    # e_p: embedding for the concept prompt to be erased
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
        neg_prompts, text_encoders, tokenizers
    )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )
    
    # Sample random timestep for training (ZImage uses 9 steps for inference)
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    # Map to actual timestep range [0, 1000]
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)
    
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    
    # Prepare guidance (ZImage guidance mechanism)
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    with torch.no_grad():
        # Clear cache before latent_sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Generate noisy latent with the concept prompt using ZImage model
        # This samples up to timestep t_enc using the concept embedding
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1], 
                                            512,
                                            512,
                                            emb_p.to(transformer.device),
                                            pooled_emb_p.to(transformer.device),
                                            text_ids_p.to(transformer.device),
                                            start_guidance, 
                                            int(ddim_steps),
                                            vae_scale_factor
                                        )
        # e_0: Predict noise with negative (unconditional) embedding
        e_0 = predict_noise(
            transformer, z, emb_0, pooled_emb_0, text_ids_0, 
            latent_image_ids, guidance=start_guidance, 
            timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
        )
        
        # e_p: Predict noise with concept embedding (to be erased)
        e_p = predict_noise(
            transformer, z, emb_p, pooled_emb_p, text_ids_p, 
            latent_image_ids, guidance=start_guidance, 
            timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
        )
        # Release embeddings after reference predictions
        del emb_0, pooled_emb_0, text_ids_0
    
    # e_n: Get conditional score from LoRA-adapted model (trainable)
    e_n = predict_noise(
        transformer, z, emb_p, pooled_emb_p, text_ids_p, 
        latent_image_ids, guidance=start_guidance, 
        timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
    )
    
    # Freeze the reference predictions from pretrained model
    e_0.requires_grad = False
    e_p.requires_grad = False
    
    # ESD Loss: Push model prediction away from concept
    # Target: e_0 - negative_guidance * (e_p - e_0)
    # This moves the prediction away from e_p (concept) towards e_0 (unconditional)
    loss_esd = criteria(
        e_n.to(transformer.device), 
        e_0.to(transformer.device) - (negative_guidance * (e_p.to(transformer.device) - e_0.to(transformer.device)))
    )
    
    # Release intermediate variables after loss calculation
    del e_0, e_p, z, latent_image_ids
    
    return loss_esd, t_enc_ddpm


def calculate_erase_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, 
                         transformer, noise_scheduler, prompts, vae, criteria, 
                         negative_guidance, weight_dtype, neg_prompts, start_guidance=0, 
                         ddim_steps=9, latents_cache=None, step=0):
    """
    Interpolation-style erase loss: force outputs toward the negative prompt prediction.
    """
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels

    if args.cache_latents and latents_cache is not None:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)

    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
        neg_prompts, text_encoders, tokenizers
    )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )

    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)

    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])

    with torch.no_grad():
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1],
                                            512,
                                            512,
                                            emb_p.to(transformer.device),
                                            pooled_emb_p.to(transformer.device),
                                            text_ids_p.to(transformer.device),
                                            start_guidance,
                                            int(ddim_steps),
                                            vae_scale_factor,
                                        )

        e_0 = predict_noise(
            transformer, z, emb_0, pooled_emb_0, text_ids_0,
            latent_image_ids, guidance=start_guidance,
            timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
        )

    e_n = predict_noise(
        transformer, z, emb_p, pooled_emb_p, text_ids_p,
        latent_image_ids, guidance=start_guidance,
        timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
    )

    e_0.requires_grad = False
    loss_interp = criteria(
        e_n.to(transformer.device),
        e_0.to(transformer.device)
    )

    return loss_interp, t_enc_ddpm


def calculate_ca_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, 
                      transformer, noise_scheduler, prompts, vae, criteria, 
                      negative_guidance, weight_dtype, neg_prompts, start_guidance=0, 
                      ddim_steps=9, latents_cache=None, step=0):
    """
    @date: 2025.12.12 - 12.20 (FIXED)
    @func: CA loss adapted from FLUX to ZImage architecture
    
    CRITICAL FIX: Only collect attention maps during concept prompt forward,
    NOT during neutral prompt forward. This ensures remove_indices correctly
    index the target tokens in the concept prompt.
    """
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Import attention collection control functions
    try:
        from utils.zimage_text_lora import enable_attention_collection, disable_attention_collection
        has_collection_control = True
    except ImportError:
        has_collection_control = False
    
    # Convert images to latent space
    if args.cache_latents and latents_cache is not None:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    # Get embeddings: prompts as ca_prompt_p, neg_prompts as ca_prompt_0
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
        neg_prompts, text_encoders, tokenizers
    )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )
    
    # Sample random timestep
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)
    
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    
    # Prepare guidance
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    # Generate noisy latent with concept prompt
    with torch.no_grad():
        # Clear cache before latent_sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1], 
                                            512,
                                            512,
                                            emb_p.to(transformer.device),
                                            pooled_emb_p.to(transformer.device),
                                            text_ids_p.to(transformer.device),
                                            start_guidance, 
                                            int(ddim_steps),
                                            vae_scale_factor
                                        )
        # Release memory immediately after latent_sample
    
    # === FORWARD 1: Concept prompt (ENABLE attention collection) ===
    if has_collection_control:
        enable_attention_collection(transformer)
    
    # Predict noise with concept prompt (trainable, with LoRA)
    # Attention maps will be collected during this forward
    model_pred_p = predict_noise(
        transformer, z, emb_p, pooled_emb_p, text_ids_p, 
        latent_image_ids, guidance=start_guidance, 
        timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
    )
    
    # === FORWARD 2: Neutral prompt (DISABLE attention collection) ===
    if has_collection_control:
        disable_attention_collection(transformer)
    
    # Predict noise with negative prompt (stop gradient)
    # NO attention maps should be collected during this forward
    with torch.no_grad():
        model_pred_0 = predict_noise(
            transformer, z, emb_0, pooled_emb_0, text_ids_0, 
            latent_image_ids, guidance=start_guidance, 
            timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
        )
        # Release memory for reference prediction
        del emb_0, pooled_emb_0, text_ids_0
    
    # Re-enable collection for next iteration
    if has_collection_control:
        enable_attention_collection(transformer)
    
    # CA loss: MSE between predictions with different prompts
    loss_ca = torch.mean(
        ((model_pred_p.float() - model_pred_0.float()) ** 2).reshape(model_pred_0.shape[0], -1),
        1,
    )[0]
    
    # Release intermediate variables after loss calculation
    del model_pred_0, z, latent_image_ids
    
    return loss_ca, t_enc_ddpm


def calculate_lial_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, 
                        transformer, noise_scheduler, prompts, vae, criteria, 
                        negative_guidance, weight_dtype, neg_prompts, start_guidance=0, 
                        ddim_steps=9, latents_cache=None, step=0):
    """
    @date: 2025.12.14
    @func: LIAL (Latent Injection Alignment Loss) - Structured velocity alignment
    """
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Localized layers for intervention (based on knowledge localization results)
    intervention_layers = [19, 22, 21, 20, 17, 16]
    
    # Convert images to latent space
    if args.cache_latents and latents_cache is not None:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    # E_anchor: anchor/neutral prompt (neg_prompts, e.g., "a person")
    # E_target: target prompt with concept (prompts, e.g., "a nude person")
    emb_anchor, pooled_emb_anchor, text_ids_anchor = compute_text_embeddings(
        neg_prompts, text_encoders, tokenizers
    )
    emb_target, pooled_emb_target, text_ids_target = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )
    
    # Sample random timestep
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)
    
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    
    # Prepare guidance
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    # Generate noisy latent with target prompt (for both v_ref and v_rep)
    with torch.no_grad():
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1], 
                                            512,
                                            512,
                                            emb_target.to(transformer.device),
                                            pooled_emb_target.to(transformer.device),
                                            text_ids_target.to(transformer.device),
                                            start_guidance, 
                                            int(ddim_steps),
                                            vae_scale_factor
                                        )
    
    # Step 1: Compute v_ref = v_θ(x_t, E_anchor, t)
    # Standard forward with anchor prompt (no replacement)
    with torch.no_grad():
        v_ref = predict_noise(
            transformer, z, emb_anchor, pooled_emb_anchor, text_ids_anchor, 
            latent_image_ids, guidance=start_guidance, 
            timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
        )
        v_ref.requires_grad = False
    
    # Step 2: Compute v_rep = v_θ^{R_L_κ}(x_t, E_target, t)
    # Forward with target prompt BUT replace text embeddings in L_κ layers
    
    # Save original processors
    original_processors = {}
    for idx, layer in enumerate(transformer.layers):
        original_processors[idx] = layer.attention.processor
    
    try:
        # Pass 1: Set processors to SAVE mode and run with anchor prompt
        # This saves anchor text embeddings in intervention layers
        for idx, layer in enumerate(transformer.layers):
            if idx in intervention_layers:
                processor = ZImageEmbeddingReplacementProcessor(
                    mode=ZImageEmbeddingReplacementProcessor.ProcessorMode.SAVE_TEXT_HIDDEN_STATES,
                    image_seq_len=1024
                )
            else:
                processor = ZImageEmbeddingReplacementProcessor(
                    mode=ZImageEmbeddingReplacementProcessor.ProcessorMode.NONE,
                    image_seq_len=1024
                )
            layer.attention.set_processor(processor)
        
        # Run forward with anchor to save embeddings
        with torch.no_grad():
            _ = predict_noise(
                transformer, z, emb_anchor, pooled_emb_anchor, text_ids_anchor, 
                latent_image_ids, guidance=start_guidance, 
                timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
            )
        
        # Pass 2: Set processors to REPLACE mode and run with target prompt
        # This creates the structured velocity v_rep
        for idx, layer in enumerate(transformer.layers):
            if idx in intervention_layers:
                # Create new processor in REPLACE mode
                processor = ZImageEmbeddingReplacementProcessor(
                    mode=ZImageEmbeddingReplacementProcessor.ProcessorMode.REPLACE_TEXT_HIDDEN_STATES,
                    image_seq_len=1024
                )
                # Transfer saved embeddings from SAVE processor
                old_processor = layer.attention.processor
                if hasattr(old_processor, 'saved_text_hidden_states'):
                    processor.saved_text_hidden_states = old_processor.saved_text_hidden_states
            else:
                processor = ZImageEmbeddingReplacementProcessor(
                    mode=ZImageEmbeddingReplacementProcessor.ProcessorMode.NONE,
                    image_seq_len=1024
                )
            layer.attention.set_processor(processor)
        
        # Now run with target prompt - this is v_rep (trainable)
        # The model sees E_target as input, but E_anchor in intervention layers
        v_rep = predict_noise(
            transformer, z, emb_target, pooled_emb_target, text_ids_target, 
            latent_image_ids, guidance=start_guidance, 
            timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
        )
    
    finally:
        # Restore original processors
        for idx, layer in enumerate(transformer.layers):
            layer.attention.set_processor(original_processors[idx])
    
    # Compute LIAL loss: align structured velocity with reference
    loss_lial = torch.mean(
        ((v_rep.float() - v_ref.float()) ** 2).reshape(v_ref.shape[0], -1),
        1,
    )[0]
    
    return loss_lial, t_enc_ddpm


def calculate_preserve_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, 
                            transformer, noise_scheduler, prompts, vae, criteria, 
                            negative_guidance, weight_dtype, neg_prompts, start_guidance=0, 
                            ddim_steps=9, latents_cache=None, step=0):
    """
    @date: 2025.12.13
    @func: Preserve loss to maintain model performance on non-target concepts
           L_preserve = || v(x_t, c_preserve) - v0(x_t, c_preserve) ||^2
           where c_preserve uses neg_prompts
    """
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents and latents_cache is not None:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    # Get embeddings for preserve prompt (neg_prompts)
    emb_preserve, pooled_emb_preserve, text_ids_preserve = compute_text_embeddings(
        neg_prompts, text_encoders, tokenizers
    )
    
    # Sample random timestep
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)
    
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    
    # Prepare guidance
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    # Generate noisy latent with preserve prompt
    with torch.no_grad():
        # Clear cache before latent_sample to free up memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1], 
                                            512,
                                            512,
                                            emb_preserve.to(transformer.device),
                                            pooled_emb_preserve.to(transformer.device),
                                            text_ids_preserve.to(transformer.device),
                                            start_guidance, 
                                            int(ddim_steps),
                                            vae_scale_factor
                                        )
        # v0: Predict noise with pretrained model (disable LoRA to get baseline)
        # Check if using PEFT adapters or custom LoRA
        # Try to use PEFT disable_adapters if available, otherwise use custom method
        try:
            # Standard PEFT adapter - can disable/enable
            transformer.disable_adapters()
            v0 = predict_noise(
                transformer, z, emb_preserve, pooled_emb_preserve, text_ids_preserve, 
                latent_image_ids, guidance=start_guidance, 
                timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
            )
            transformer.enable_adapters()
        except (ValueError, AttributeError):
            # Custom position-masked LoRA - temporarily zero out LoRA weights
            # Save current LoRA weights
            lora_weights = {}
            for idx, layer in enumerate(transformer.layers):
                attn = layer.attention
                if hasattr(attn.to_q, 'lora_down') and hasattr(attn.to_q, 'lora_up'):
                    lora_weights[f'q_{idx}_down'] = attn.to_q.lora_down.weight.data.clone()
                    lora_weights[f'q_{idx}_up'] = attn.to_q.lora_up.weight.data.clone()
                    attn.to_q.lora_down.weight.data.zero_()
                    attn.to_q.lora_up.weight.data.zero_()
                if hasattr(attn.to_k, 'lora_down') and hasattr(attn.to_k, 'lora_up'):
                    lora_weights[f'k_{idx}_down'] = attn.to_k.lora_down.weight.data.clone()
                    lora_weights[f'k_{idx}_up'] = attn.to_k.lora_up.weight.data.clone()
                    attn.to_k.lora_down.weight.data.zero_()
                    attn.to_k.lora_up.weight.data.zero_()
            
            # Predict with zero LoRA (baseline)
            v0 = predict_noise(
                transformer, z, emb_preserve, pooled_emb_preserve, text_ids_preserve, 
                latent_image_ids, guidance=start_guidance, 
                timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
            )
            
            # Restore LoRA weights
            for idx, layer in enumerate(transformer.layers):
                attn = layer.attention
                if hasattr(attn.to_q, 'lora_down') and f'q_{idx}_down' in lora_weights:
                    attn.to_q.lora_down.weight.data = lora_weights[f'q_{idx}_down']
                    attn.to_q.lora_up.weight.data = lora_weights[f'q_{idx}_up']
                if hasattr(attn.to_k, 'lora_down') and f'k_{idx}_down' in lora_weights:
                    attn.to_k.lora_down.weight.data = lora_weights[f'k_{idx}_down']
                    attn.to_k.lora_up.weight.data = lora_weights[f'k_{idx}_up']
            
            # Release lora_weights dict
            del lora_weights
        
        # Release memory after baseline prediction
    
    # v: Predict noise with LoRA-adapted model (trainable)
    v = predict_noise(
        transformer, z, emb_preserve, pooled_emb_preserve, text_ids_preserve, 
        latent_image_ids, guidance=start_guidance, 
        timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
    )
    
    # Freeze the reference prediction from pretrained model
    v0.requires_grad = False
    
    # Preserve loss: keep model behavior consistent on preserve concepts
    loss_preserve = criteria(
        v.to(transformer.device), 
        v0.to(transformer.device)
    )
    
    # Release intermediate variables after loss calculation
    del v0, z, latent_image_ids, emb_preserve, pooled_emb_preserve, text_ids_preserve
    
    return loss_preserve, t_enc_ddpm


def calculate_attn_loss(args, batch, compute_text_embeddings, text_encoders, tokenizers, 
                        transformer, noise_scheduler, prompts, vae, criteria, 
                        negative_guidance, weight_dtype, neg_prompts, start_guidance=0, 
                        ddim_steps=9, latents_cache=None, step=0):
    """
    @date: 2025.12.14
    @func: Attention loss for ZImage - minimizes attention from image tokens to target text tokens
           
    ZImage sequence: [image_tokens (1024), text_tokens (variable)]
    Goal: Minimize attention_maps[:, :, :1024, remove_indices] to prevent image from attending to target concept
    """
    
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Convert images to latent space
    if args.cache_latents and latents_cache is not None:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        model_input = vae.encode(pixel_values).latent_dist.sample()
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    # Get text embeddings for the concept prompt
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )
    
    # Sample random timestep
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)
    
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    
    # Prepare guidance
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    # Generate noisy latent with concept prompt
    with torch.no_grad():
        z, latent_image_ids = latent_sample(transformer,
                                            noise_scheduler,
                                            1,
                                            model_input.shape[1], 
                                            512,
                                            512,
                                            emb_p.to(transformer.device),
                                            pooled_emb_p.to(transformer.device),
                                            text_ids_p.to(transformer.device),
                                            start_guidance, 
                                            int(ddim_steps),
                                            vae_scale_factor
                                        )
        # Release memory immediately after latent_sample
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save original processors
    original_processors = []
    for layer in transformer.layers:
        original_processors.append(layer.attention.processor)
        # Install collector processor
        collector = ZImageAttnMapCollectorProcessor(image_seq_len=1024)
        layer.attention.processor = collector
    
    # Forward pass with attention map collection
    model_pred = predict_noise(
        transformer, z, emb_p, pooled_emb_p, text_ids_p, 
        latent_image_ids, guidance=start_guidance, 
        timesteps=t_enc_ddpm.to(transformer.device), CPU_only=True
    )
    
    all_attn_maps = []
    for layer in transformer.layers:
        processor = layer.attention.processor
        if isinstance(processor, ZImageAttnMapCollectorProcessor):
            if len(processor.attention_maps) > 0:
                # Each processor stores attention maps from multiple forward calls (timesteps)
                # We concatenate them: [batch, heads, image_tokens, text_tokens]
                layer_attn = torch.cat(processor.attention_maps, dim=0)
                all_attn_maps.append(layer_attn)
    
    # Restore original processors
    for i, layer in enumerate(transformer.layers):
        if i < len(original_processors):
            layer.attention.processor = original_processors[i]
    
    if len(all_attn_maps) == 0:
        raise ValueError("No attention maps collected")
    
    # Stack all layer attention maps: [num_layers * batch, heads, image_tokens, text_tokens]
    stacked_attn_maps = torch.cat(all_attn_maps, dim=0)
    
    # Get remove_indices (target token positions in text sequence)
    remove_indices = batch['remove_indices'][0]
    
    if remove_indices is None or len(remove_indices) == 0:
        raise ValueError("No target tokens to erase")
    
    # Create attention mask: we want to minimize attention at remove_indices positions
    # attn_maps shape: [batch*layers, heads, image_tokens, text_tokens]
    # We want to zero out positions [..., remove_indices] in the text dimension
    
    # Extract attention to target tokens: [batch*layers, heads, image_tokens, len(remove_indices)]
    target_attn = stacked_attn_maps[:, :, :, remove_indices]
    
    # Calculate loss: minimize L2 norm of attention to target tokens
    # We want to minimize attention to remove_indices positions
    loss_attn = torch.norm(target_attn, p=2)
    
    # Release intermediate variables after loss calculation
    del model_pred, all_attn_maps, stacked_attn_maps, z, latent_image_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return loss_attn, t_enc_ddpm