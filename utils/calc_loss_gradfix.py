# coding: UTF-8
"""
@date: 2025.12.20
@func: Gradient-preserving loss calculation for ZImage concept erasure

ROOT CAUSE FIX: The original calc_loss.py wraps everything in torch.no_grad(),
which breaks the gradient graph for training. This version carefully preserves
gradients where needed.
"""

import torch
import torch.nn.functional as F
from typing import List, Optional

try:
    from .esd_utils import latent_sample, zimage_pack_latents, _prepare_latent_image_ids
except ImportError:
    from esd_utils import latent_sample, zimage_pack_latents, _prepare_latent_image_ids


def predict_noise_with_grad(
    transformer,
    latent_code,
    prompt_embeds,
    pooled_prompt_embeds,
    text_ids,
    latent_image_ids,
    guidance,
    timesteps,
):
    """
    This is the gradient-preserving version of predict_noise
    Key: No @torch.no_grad() decorator!
    """
    device = transformer.device
    
    latent_code_tensor = latent_code.to(device)
    
    batch_size = latent_code_tensor.shape[0]
    
    # Convert to list format with proper 4D shape (C, F, H, W)
    latent_code_list = []
    for b in range(batch_size):
        latent_4d = latent_code_tensor[b].unsqueeze(1).to(torch.bfloat16)
        latent_code_list.append(latent_4d)
    
    # Handle prompt_embeds: can be tensor or list
    if isinstance(prompt_embeds, list):
        # Already a list, just move to device
        cap_feats_list = [pe.to(device) if isinstance(pe, torch.Tensor) else pe for pe in prompt_embeds]
    elif isinstance(prompt_embeds, torch.Tensor):
        # Convert tensor to list format
        prompt_embeds_tensor = prompt_embeds.to(device)
        cap_feats_list = [prompt_embeds_tensor[b] for b in range(batch_size)]
    else:
        # Fallback: try to convert
        cap_feats_list = [pe.to(device) for pe in prompt_embeds]
    
    # Handle timesteps
    if isinstance(timesteps, torch.Tensor):
        t_input = timesteps.to(device)
        if t_input.dim() == 0:
            t_input = t_input.unsqueeze(0)
        if t_input.shape[0] == 1 and batch_size > 1:
            t_input = t_input.expand(batch_size)
    else:
        t_input = torch.tensor([timesteps], device=device)
        if batch_size > 1:
            t_input = t_input.expand(batch_size)
    
    output = transformer(
        latent_code_list,
        t_input,
        cap_feats_list,
    )
    
    # Handle ZImage transformer output format
    if isinstance(output, list):
        model_pred = torch.stack([out.squeeze(1) for out in output], dim=0)
    elif isinstance(output, tuple):
        if isinstance(output[0], list):
            model_pred = torch.stack([out.squeeze(1) for out in output[0]], dim=0)
        else:
            model_pred = output[0].squeeze(1) if output[0].dim() == 5 else output[0]
    else:
        model_pred = output.squeeze(1) if output.dim() == 5 else output
    
    # Release intermediate variables (but keep model_pred for gradient computation)
    # Note: We can't delete latent_code_tensor if it's needed for gradients, but we can delete the list copies
    del latent_code_list, cap_feats_list, t_input
    if isinstance(prompt_embeds, torch.Tensor):
        del prompt_embeds_tensor
    
    return model_pred


def calculate_ca_loss_with_grad(
    args, batch, compute_text_embeddings, text_encoders, tokenizers,
    transformer, noise_scheduler, prompts, vae, criteria,
    negative_guidance, weight_dtype, neg_prompts, start_guidance=0,
    ddim_steps=9, latents_cache=None, step=0
):
    """
    CA loss with PROPER GRADIENT FLOW
    
    Critical fix: Only use no_grad for reference predictions, keep grad for trainable
    """
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Import attention collection control
    try:
        from utils.zimage_text_lora import enable_attention_collection, disable_attention_collection
        has_collection_control = True
    except ImportError:
        has_collection_control = False
    
    # Get latent
    if args.cache_latents and latents_cache is not None:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        with torch.no_grad():  # VAE encoding doesn't need grad
            model_input = vae.encode(pixel_values).latent_dist.sample()
    
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    # Get embeddings
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
        neg_prompts, text_encoders, tokenizers
    )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )
    
    # Sample timestep
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)
    
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    # Generate noisy latent (this part doesn't need grad)
    # Handle emb_p format: can be tensor or list
    if isinstance(emb_p, list):
        emb_p_for_sample = emb_p  # latent_sample can handle list
    else:
        emb_p_for_sample = emb_p.to(transformer.device)
    
    if isinstance(pooled_emb_p, torch.Tensor):
        pooled_emb_p_for_sample = pooled_emb_p.to(transformer.device)
    else:
        pooled_emb_p_for_sample = pooled_emb_p
    
    if isinstance(text_ids_p, torch.Tensor):
        text_ids_p_for_sample = text_ids_p.to(transformer.device)
    else:
        text_ids_p_for_sample = text_ids_p
    
    with torch.no_grad():
        z, latent_image_ids = latent_sample(
            transformer, noise_scheduler, 1,
            model_input.shape[1], 512, 512,
            emb_p_for_sample,
            pooled_emb_p_for_sample,
            text_ids_p_for_sample,
            start_guidance, int(ddim_steps), vae_scale_factor
        )
        # Release memory immediately after latent_sample
        del emb_p_for_sample, pooled_emb_p_for_sample, text_ids_p_for_sample
    
    # Enable collection for concept forward
    if has_collection_control:
        enable_attention_collection(transformer)
    
    # No torch.no_grad() here!
    model_pred_p = predict_noise_with_grad(
        transformer, z, emb_p, pooled_emb_p, text_ids_p,
        latent_image_ids, guidance=start_guidance,
        timesteps=t_enc_ddpm.to(transformer.device)
    )
    
    # Disable collection for neutral forward
    if has_collection_control:
        disable_attention_collection(transformer)
    
    with torch.no_grad():
        model_pred_0 = predict_noise_with_grad(
            transformer, z, emb_0, pooled_emb_0, text_ids_0,
            latent_image_ids, guidance=start_guidance,
            timesteps=t_enc_ddpm.to(transformer.device)
        )
        # Release memory for reference prediction
        del emb_0, pooled_emb_0, text_ids_0
    
    # Re-enable for next iteration
    if has_collection_control:
        enable_attention_collection(transformer)
    
    # CA loss: MSE between predictions
    # model_pred_p has grad, model_pred_0 doesn't
    loss_ca = torch.mean(
        ((model_pred_p.float() - model_pred_0.float()) ** 2).reshape(model_pred_0.shape[0], -1),
        1,
    )[0]
    
    # Release intermediate variables after loss calculation
    del model_pred_0, z, latent_image_ids
    
    return loss_ca, t_enc_ddpm


def calculate_esd_loss_with_grad(
    args, batch, compute_text_embeddings, text_encoders, tokenizers,
    transformer, noise_scheduler, prompts, vae, criteria,
    negative_guidance, weight_dtype, neg_prompts, start_guidance=0,
    ddim_steps=9, latents_cache=None, step=0
):
    """
    ESD loss with proper gradient flow
    """
    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    # Get latent
    if args.cache_latents and latents_cache is not None:
        model_input = latents_cache[step].sample()
    else:
        pixel_values = batch["pixel_values"].to(dtype=vae.dtype).cuda()
        with torch.no_grad():
            model_input = vae.encode(pixel_values).latent_dist.sample()
    
    model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
    model_input = model_input.to(dtype=weight_dtype)
    
    # Get embeddings
    emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings(
        neg_prompts, text_encoders, tokenizers
    )
    emb_p, pooled_emb_p, text_ids_p = compute_text_embeddings(
        prompts, text_encoders, tokenizers
    )
    
    # Sample timestep
    t_enc = torch.randint(ddim_steps, (1,), device=transformer.device)
    og_num = round((int(t_enc) / ddim_steps) * 1000)
    og_num_lim = round((int(t_enc + 1) / ddim_steps) * 1000)
    t_enc_ddpm = torch.randint(og_num, og_num_lim, (1,), device=transformer.device)
    
    vae_scale_factor = 2 ** (len(vae_config_block_out_channels))
    start_guidance = torch.tensor([start_guidance], device=transformer.device)
    start_guidance = start_guidance.expand(model_input.shape[0])
    
    # Generate noisy latent
    # Handle emb_p format: can be tensor or list
    if isinstance(emb_p, list):
        emb_p_for_sample = emb_p  # latent_sample can handle list
    else:
        emb_p_for_sample = emb_p.to(transformer.device)
    
    if isinstance(pooled_emb_p, torch.Tensor):
        pooled_emb_p_for_sample = pooled_emb_p.to(transformer.device)
    else:
        pooled_emb_p_for_sample = pooled_emb_p
    
    if isinstance(text_ids_p, torch.Tensor):
        text_ids_p_for_sample = text_ids_p.to(transformer.device)
    else:
        text_ids_p_for_sample = text_ids_p
    
    with torch.no_grad():
        z, latent_image_ids = latent_sample(
            transformer, noise_scheduler, 1,
            model_input.shape[1], 512, 512,
            emb_p_for_sample,
            pooled_emb_p_for_sample,
            text_ids_p_for_sample,
            start_guidance, int(ddim_steps), vae_scale_factor
        )
        # Release memory immediately after latent_sample
        del emb_p_for_sample, pooled_emb_p_for_sample, text_ids_p_for_sample
        
        # Reference predictions (stop gradient)
        e_0 = predict_noise_with_grad(
            transformer, z, emb_0, pooled_emb_0, text_ids_0,
            latent_image_ids, guidance=start_guidance,
            timesteps=t_enc_ddpm.to(transformer.device)
        )
        
        e_p = predict_noise_with_grad(
            transformer, z, emb_p, pooled_emb_p, text_ids_p,
            latent_image_ids, guidance=start_guidance,
            timesteps=t_enc_ddpm.to(transformer.device)
        )
        # Release embeddings after reference predictions
        del emb_0, pooled_emb_0, text_ids_0
    
    # Trainable prediction (WITH GRADIENT)
    e_n = predict_noise_with_grad(
        transformer, z, emb_p, pooled_emb_p, text_ids_p,
        latent_image_ids, guidance=start_guidance,
        timesteps=t_enc_ddpm.to(transformer.device)
    )
    
    # ESD Loss
    loss_esd = criteria(
        e_n.to(transformer.device),
        e_0.to(transformer.device) - (negative_guidance * (e_p.to(transformer.device) - e_0.to(transformer.device)))
    )
    
    # Release intermediate variables after loss calculation
    del e_0, e_p, z, latent_image_ids
    
    return loss_esd, t_enc_ddpm

