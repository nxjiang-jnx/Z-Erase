import torch
from diffusers.utils.torch_utils import randn_tensor
from diffusers import ZImagePipeline
from typing import Any, Callable, Dict, List, Optional, Union


def calculate_shift(
    image_seq_len,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
):
    """
    Calculate mu shift factor for Z-Image timestep schedule
    """
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def zimage_pack_latents(latents, batch_size, num_channels_latents, height, width):
    latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    latents = latents.reshape(batch_size, (height // 2) * (width // 2), num_channels_latents * 4)
    
    return latents


def zimage_unpack_latents(latents, height, width, vae_scale_factor):
    batch_size, num_patches, channels = latents.shape

    height = height // vae_scale_factor
    width = width // vae_scale_factor

    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    latents = latents.reshape(batch_size, channels // (2 * 2), height, width)

    return latents


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]

    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape

    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )

    return latent_image_ids.to(device=device, dtype=dtype)


@torch.no_grad()
def latent_sample(transformer, scheduler, batch_size, num_channels_latents, height, width, prompt_embeds, pooled_prompt_embeds, text_ids, guidance, timesteps, vae_scale_factor, latents=None, return_attn=False, mu=None):
    """
        Sample the model
        ESD quick_sample_till_t
    """
    height = int(height) // 8  # vae_scale_factor
    width = int(width) // 8
    shape = (batch_size, num_channels_latents, height, width)
    
    # (A) Generate random tensor
    if latents is None:
        latents = randn_tensor(shape, generator=None, dtype=torch.bfloat16)
    
    # ZImage expects 4D input (C, F, H, W) where F=1 for images
    latents = latents.to(transformer.device).bfloat16()
    
    # Prepare latent_image_ids (not used by ZImage but keep for compatibility)
    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, transformer.device, torch.bfloat16)
    
    # Prepare text embeddings
    scheduler.set_train_timesteps(timesteps, device=transformer.device)
    timesteps = scheduler.timesteps
    
    pooled_prompt_embeds = pooled_prompt_embeds.bfloat16()
    prompt_embeds = prompt_embeds.bfloat16()
    text_ids = text_ids.bfloat16()
    
    attn_map_lst = []
    
    # Denoising loop
    for i, t in enumerate(timesteps):
        # Broadcast to batch dimension
        timestep = t.expand(batch_size).to(torch.bfloat16)
        
        # ZImage transformer forward call
        latents_list = []
        for b in range(batch_size):
            latent_4d = latents[b].unsqueeze(1).to(torch.bfloat16)  # (C, 1, H, W)
            latents_list.append(latent_4d)
        
        # prompt_embeds is list format for ZImage
        if isinstance(prompt_embeds, torch.Tensor):
            cap_feats_list = [prompt_embeds[b].to(torch.bfloat16) for b in range(prompt_embeds.shape[0])]
        else:
            cap_feats_list = [pe.to(torch.bfloat16) for pe in prompt_embeds]
        
        output = transformer(
            latents_list,
            timestep,
            cap_feats_list,
        )
        
        # Handle ZImage transformer output format
        if isinstance(output, list):
            noise_pred = torch.stack([out.squeeze(1) for out in output], dim=0)
            attn_maps = None
        elif isinstance(output, tuple):
            if isinstance(output[0], list):
                noise_pred = torch.stack([out.squeeze(1) for out in output[0]], dim=0)
            else:
                noise_pred = output[0].squeeze(1) if output[0].dim() == 5 else output[0]
            attn_maps = output[1] if len(output) > 1 else None
        else:
            noise_pred = output.squeeze(1) if output.dim() == 5 else output
            attn_maps = None
        
        # Compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
        
        attn_map_lst.append(attn_maps) 
        
    if return_attn:
        return latents, latent_image_ids, attn_map_lst
    else:
        return latents, latent_image_ids


def predict_noise(transformer, latent_code, prompt_embeds, pooled_prompt_embeds, 
                  text_ids, latent_image_ids, guidance, timesteps, CPU_only=False):
    """
        ESD (apply_model)
    """
    
    if CPU_only:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cuda:0")
    
    latent_code_tensor = latent_code.to(device)
    prompt_embeds_tensor = prompt_embeds.to(device)
    
    batch_size = latent_code_tensor.shape[0]
    
    # Convert to list format with proper 4D shape (C, F, H, W)
    latent_code_list = []
    for b in range(batch_size):
        latent_4d = latent_code_tensor[b].unsqueeze(1).to(torch.bfloat16)  # (C, 1, H, W)
        latent_code_list.append(latent_4d)
    
    # Convert prompt_embeds to list format (one tensor per batch item)
    cap_feats_list = [prompt_embeds_tensor[b] for b in range(batch_size)]
    
    # Handle timesteps: ensure it's a tensor with proper shape
    if isinstance(timesteps, torch.Tensor):
        t_input = timesteps.to(device)
        # If timesteps is a scalar or 1D, expand to batch size
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
        t_input,  # t is passed directly, transformer handles scaling internally
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
    
    return model_pred
