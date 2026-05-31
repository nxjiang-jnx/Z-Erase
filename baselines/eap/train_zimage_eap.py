"""
@date: 2025.12.28
@func: EAP (NeurIPS 2024) adapted for ZImage-Turbo
"""

# Need 2 GPUs:
# GPU0 for frozen model, 20G is enough for frozen model
# GPU1 for training model, you need at least 96G VRAM!
# Train specific layers may save some VRAM

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["CUDA_VISIBLE_DEVICES"] = "4,6"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from diffusers import ZImagePipeline
from diffusers.utils.torch_utils import randn_tensor
from gen_embedding_zimage import (
    create_prompt,
    learn_k_means_from_input_embedding,
    save_embedding_matrix,
    search_closest_tokens,
)
from PIL import Image
from safetensors.torch import save_file
from torch.autograd import Variable
from torchvision import transforms
from tqdm import tqdm
from utils_zimage import gumbel_softmax, save_to_dict, zimage_pack_latents, zimage_unpack_latents

# Import ZImage's position-masked LoRA
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
from zimage_text_lora import (
    apply_text_masked_lora_to_transformer,
    enable_lora,
    disable_lora,
)


def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
    """Prepare latent image IDs (for positional encoding)"""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
    latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
    
    latent_image_id_height, latent_image_id_width, latent_image_id_channels = latent_image_ids.shape
    
    latent_image_ids = latent_image_ids.reshape(
        latent_image_id_height * latent_image_id_width, latent_image_id_channels
    )
    
    return latent_image_ids.to(device=device, dtype=dtype)


@torch.no_grad()
def latent_sample(
    transformer,
    scheduler,
    batch_size,
    num_channels_latents,
    height,
    width,
    prompt_embeds,
    pooled_prompt_embeds,
    text_ids,
    timesteps,
):
    """
    Sample latent for ZImage
    Similar to quick_sample_till_t of ESD
    """
    height = int(height) // 8  # vae_scale_factor
    width = int(width) // 8
    shape = (batch_size, num_channels_latents, height, width)
    
    # (A) Generate random tensor (4D format: [B, C, H, W])
    latents = randn_tensor(shape, generator=None, dtype=torch.bfloat16)
    latents = latents.to(transformer.device).bfloat16()
    
    # Prepare latent_image_ids (not used by ZImage but keep for compatibility)
    latent_image_ids = _prepare_latent_image_ids(batch_size, height // 2, width // 2, transformer.device, torch.bfloat16)
    
    # (B) Generate latents
    scheduler.set_timesteps(timesteps, device=transformer.device)
    timesteps_list = scheduler.timesteps
    
    prompt_embeds = prompt_embeds.bfloat16()
    
    # Denoising loop
    for i, t in enumerate(timesteps_list):
        # Broadcast to batch dimension
        timestep = t.expand(batch_size).to(torch.bfloat16)
        # Normalize timestep: (1000 - t) / 1000
        timestep_norm = (1000 - timestep) / 1000
        
        # ZImage transformer expects list format
        # Convert latents to list of 4D tensors: [C, 1, H, W]
        latents_list = []
        for b in range(batch_size):
            latent_4d = latents[b].unsqueeze(1).to(torch.bfloat16)  # (C, 1, H, W)
            latents_list.append(latent_4d)
        
        # Convert prompt_embeds to list format
        if isinstance(prompt_embeds, torch.Tensor):
            cap_feats_list = [prompt_embeds[b].to(torch.bfloat16) for b in range(prompt_embeds.shape[0])]
        else:
            cap_feats_list = [pe.to(torch.bfloat16) for pe in prompt_embeds]
        
        # ZImage transformer forward call
        output = transformer(
            latents_list,
            timestep_norm,
            cap_feats_list,
        )
        
        # Handle ZImage transformer output format
        if isinstance(output, list):
            noise_pred = torch.stack([out.squeeze(1) for out in output], dim=0)
        elif isinstance(output, tuple):
            if isinstance(output[0], list):
                noise_pred = torch.stack([out.squeeze(1) for out in output[0]], dim=0)
            else:
                noise_pred = output[0].squeeze(1) if output[0].dim() == 5 else output[0]
        else:
            noise_pred = output.squeeze(1) if output.dim() == 5 else output
        
        # Compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]
    
    return latents, latent_image_ids


def predict_noise(
    transformer,
    latent_code,
    prompt_embeds,
    text_ids,
    latent_image_ids,
    pooled_prompt_embeds,
    timesteps,
    device_id=0,
    dtype=torch.bfloat16,
):
    """
    Predict noise for ZImage
    """
    device = torch.device(f"cuda:{device_id}")
    
    # Don't move transformer every time - assume it's already on the correct device
    # transformer = transformer.to(device).to(dtype)  # Removed to avoid unnecessary moves
    
    # Convert inputs to device and dtype
    latent_code_tensor = latent_code.to(device, dtype=dtype, non_blocking=True)
    prompt_embeds_tensor = prompt_embeds.to(device, dtype=dtype, non_blocking=True)
    
    batch_size = latent_code_tensor.shape[0]
    
    # Convert to list format with proper 4D shape (C, 1, H, W)
    latent_code_list = []
    for b in range(batch_size):
        latent_4d = latent_code_tensor[b].unsqueeze(1)  # (C, 1, H, W)
        latent_code_list.append(latent_4d)
    
    # Convert prompt_embeds to list format (one tensor per batch item)
    cap_feats_list = [prompt_embeds_tensor[b] for b in range(batch_size)]
    
    # Handle timesteps: normalize and expand to batch size
    if isinstance(timesteps, torch.Tensor):
        t_input = timesteps.to(device, non_blocking=True)
        # If timesteps is a scalar or 1D, expand to batch size
        if t_input.dim() == 0:
            t_input = t_input.unsqueeze(0)
        if t_input.shape[0] == 1 and batch_size > 1:
            t_input = t_input.expand(batch_size)
    else:
        t_input = torch.tensor([timesteps], device=device, dtype=torch.bfloat16)
        if batch_size > 1:
            t_input = t_input.expand(batch_size)
    
    # Normalize timestep: (1000 - t) / 1000
    t_input_norm = (1000 - t_input) / 1000
    
    # ZImage transformer forward call
    output = transformer(
        latent_code_list,
        t_input_norm,
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
    
    # Clean up intermediate variables to free memory immediately
    del latent_code_list, cap_feats_list, t_input, t_input_norm
    del latent_code_tensor, prompt_embeds_tensor
    if isinstance(output, (list, tuple)):
        del output
    
    return model_pred


def compute_text_embeddings_zimage(prompt, pipe):
    """
    Compute ZImage's text embedding
    """
    with torch.no_grad():
        text_inputs = pipe.tokenizer(
            [prompt] if isinstance(prompt, str) else prompt,
            padding="max_length",
            max_length=256,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(pipe.device)
        attention_mask = text_inputs.attention_mask.to(pipe.device)
        
        outputs = pipe.text_encoder(
            input_ids=text_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        if hasattr(outputs, 'last_hidden_state'):
            prompt_embeds = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
            prompt_embeds = outputs.hidden_states[-1]
        else:
            raise ValueError("Cannot extract hidden states from text encoder")
        
        # Compute pooled embeddings
        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)
        sum_embeddings = torch.sum(prompt_embeds * attention_mask_expanded, dim=1)
        sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
        pooled_prompt_embeds = sum_embeddings / sum_mask
        
        # Text IDs for positional encoding
        batch_size = prompt_embeds.shape[0]
        seq_len = prompt_embeds.shape[1]
        text_ids = torch.zeros(batch_size, seq_len, 3).to(device=pipe.device, dtype=prompt_embeds.dtype)
        
        return prompt_embeds, pooled_prompt_embeds, text_ids


def moving_average(a, n=3):
    """Compute moving average"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def plot_loss(losses, path, word, n=100):
    """Plot loss curve"""
    v = moving_average(losses, n)
    plt.plot(v, label=f"{word}_loss")
    plt.legend(loc="upper left")
    plt.title("Average loss in training", fontsize=20)
    plt.xlabel("Data point", fontsize=16)
    plt.ylabel("Loss value", fontsize=16)
    plt.savefig(path)
    plt.close()


def get_models(devices):
    """
    Load ZImage model
    Use two GPUs: devices[0] for original model, devices[1] for training model
    """
    print("[Loading Models] Loading ZImage-Turbo...")
    
    device_0 = torch.device(devices[0])
    device_1 = torch.device(devices[1])
    
    # Clear both GPUs before loading models
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_0)
        torch.cuda.synchronize(device_1)
    
    # Load first model (original) - keep on CPU first
    print("[Loading Models] Loading original model...")
    zimage_orig = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", 
        torch_dtype=torch.bfloat16,
        cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    
    # Move original model to GPU 0
    model_orig = zimage_orig.transformer.to(device_0)
    
    # Clear GPU 0 cache and free zimage_orig pipeline
    del zimage_orig
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_0)
    
    # Clear GPU 1 cache before loading second model
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_1)
    
    # Load second model (training) - keep on CPU first
    print("[Loading Models] Loading training model...")
    zimage_model = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", 
        torch_dtype=torch.bfloat16,
        cache_dir=os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    )
    
    # Move pipeline to GPU 0 (for text encoder, VAE, etc.)
    zimage_model = zimage_model.to(device_0)
    
    # Move only transformer to GPU 1
    model = zimage_model.transformer.to(device_1)
    
    # Clear both GPUs after model loading
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_0)
        torch.cuda.synchronize(device_1)
    
    print(f"[Models Loaded] Original model on {devices[0]}, Training model on {devices[1]}")
    
    return zimage_model, model_orig, model


def train_eap(
    prompt,
    start_guidance,
    negative_guidance,
    iterations,
    lr,
    devices,
    output_dir="models",
    seperator=None,
    image_size=512,
    ddim_steps=9,
    gumbel_k_closest=1000,
    gumbel_num_centers=100,
    gumbel_lr=1e-3,
    gumbel_temp=2,
    gumbel_hard=1,
    lora_rank=64,
    target_lora_layers=None,
):
    """
    Train ZImage using EAP method for concept erasure
    """
    # PROMPT clean
    word_print = prompt.replace(" ", "")
    
    if seperator is not None:
        words = prompt.split(seperator)
        erased_words = [word.strip() for word in words]
    else:
        erased_words = [prompt]
    print(f"[Erasing Concepts] {erased_words}")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Model settings
    zimage_model, model_orig, model = get_models(devices)
    zimage_model.vae.enable_slicing()
    zimage_model.vae.enable_tiling()
    
    # Convert device strings to torch.device objects
    device_0 = torch.device(devices[0])
    device_1 = torch.device(devices[1])
    
    # Apply Position-Masked LoRA to training model
    print(f"[Position-Masked LoRA] Applying to training model...")
    
    # Parse target layers if provided as string
    if target_lora_layers is not None and isinstance(target_lora_layers, str):
        target_lora_layers = [int(x.strip()) for x in target_lora_layers.split(",")]
        print(f"[LoRA] Applying to layers: {target_lora_layers}")
    
    trainable_params = apply_text_masked_lora_to_transformer(
        model,
        target_layers=target_lora_layers,
        lora_rank=lora_rank,
        image_seq_len=1024,
    )
    
    trainable_count = sum(p.numel() for p in trainable_params)
    all_params = sum(p.numel() for p in model.parameters())
    print(f"[LoRA] Trainable params: {trainable_count:,} / {all_params:,} ({100 * trainable_count / all_params:.6f}%)")
    
    # Ensure model is on GPU 1 and set training mode
    model = model.to(device_1)
    
    # CRITICAL: Disable gradient checkpointing for gradient accumulation
    # Gradient checkpointing recomputes forward during backward, which conflicts with
    # our two separate backward passes, causing memory to DOUBLE instead of saving
    if hasattr(model, 'gradient_checkpointing'):
        model.gradient_checkpointing = False
        print("[Memory] Gradient checkpointing DISABLED (incompatible with gradient accumulation)")
    
    model.train()
    
    # Clear cache after model setup
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device_1)
    
    # Optimizer
    print(f"[Optimizer] Learning rate: {lr}")
    opt = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.9, 0.99), weight_decay=1e-04, eps=1e-08)
    criteria = torch.nn.MSELoss()
    
    losses, losses_onehot = [], []
    history_dict = {}
    
    name = f"ZImage-EAP-word_{word_print}-ng_{negative_guidance}-iter_{iterations}"
    
    # ==================== EAP: Adversarial prompt learning ====================
    
    # (a) Generate embedding matrix
    print("[EAP] Step 1: Generating embedding matrix...")
    if not os.path.exists("models/embedding_matrix_dict_EN3K_zimage.pt"):
        save_embedding_matrix(zimage_model, model_name="ZImage-Turbo", save_mode="dict", vocab="EN3K")
    
    if not os.path.exists("models/embedding_matrix_array_EN3K_zimage.pt"):
        save_embedding_matrix(zimage_model, model_name="ZImage-Turbo", save_mode="array", vocab="EN3K")
    
    # (b) Search similar tokens
    print("[EAP] Step 2: Searching similar tokens...")
    tokens_embedding = []
    all_sim_dict = dict()
    for word in erased_words:
        top_k_tokens, sorted_sim_dict = search_closest_tokens(
            word, zimage_model, k=gumbel_k_closest
        )
        tokens_embedding.extend(top_k_tokens)
        all_sim_dict[word] = {key: sorted_sim_dict[key] for key in top_k_tokens}
    
    # (c) Preserved set
    if gumbel_num_centers > 0:
        assert (
            gumbel_num_centers % len(erased_words) == 0
        ), "Number of centers should be divisible by number of erased words"
    preserved_dict = dict()
    
    # (d) K-means clustering
    print("[EAP] Step 3: K-means clustering...")
    for word in erased_words:
        temp = learn_k_means_from_input_embedding(
            sim_dict=all_sim_dict[word], 
            num_centers=gumbel_num_centers
        )
        preserved_dict[word] = temp
    
    history_dict = save_to_dict(preserved_dict, "preserved_set_0", history_dict)
    
    # (e) Create preserved matrix and one-hot vector
    print("[EAP] Step 4: Creating preserved matrix...")
    one_hot_dict = dict()
    preserved_matrix_dict = dict()
    embedding_shape_dict = dict()  # Store original embedding shape for reshape
    
    for erase_word in erased_words:
        preserved_set = preserved_dict[erase_word]
        pbar = tqdm(preserved_set, desc=f"Building preserved matrix for '{erase_word}'")
        for i, word in enumerate(pbar):
            if i == 0:
                preserved_matrix = create_prompt(word)
                # Store original shape: [1, seq_len, hidden_dim]
                embedding_shape = preserved_matrix.shape[1:]  # (seq_len, hidden_dim)
            else:
                preserved_matrix = torch.cat((preserved_matrix, create_prompt(word)), dim=0)
        
        # preserved_matrix: [n, seq_len, hidden_dim]
        preserved_matrix = preserved_matrix.flatten(start_dim=1)  # [n, seq_len*hidden_dim]
        one_hot = torch.zeros(
            (1, preserved_matrix.shape[0]), 
            device=device_0, 
            dtype=preserved_matrix.dtype
        )  # [1, n]
        one_hot = one_hot + 1 / preserved_matrix.shape[0]
        one_hot = Variable(one_hot, requires_grad=True)
        
        print(f"  One-hot shape: {one_hot.shape}, Preserved matrix shape: {preserved_matrix.shape}")
        print(f"  Original embedding shape: {embedding_shape}")
        
        one_hot_dict[erase_word] = one_hot
        preserved_matrix_dict[erase_word] = preserved_matrix
        embedding_shape_dict[erase_word] = embedding_shape
    
    history_dict = save_to_dict(one_hot_dict, "one_hot_dict_0", history_dict)
    
    # Optimizer for one-hot vector
    opt_one_hot = torch.optim.Adam([one_hot for one_hot in one_hot_dict.values()], lr=gumbel_lr)
    
    # ==================== Training loop ====================
    
    pgd_num_steps = 2  # Two-stage alternating training
    pbar = tqdm(range(iterations * pgd_num_steps), desc="Training")
    
    num_channels_latents = 16  # ZImage uses 16 channels
    
    for i in pbar:
        # Clean GPU 1 cache at the start of each iteration to prevent OOM
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_1)
        
        word = random.sample(erased_words, 1)[0]
        
        # Get text embeddings
        emb_0, pooled_emb_0, text_ids_0 = compute_text_embeddings_zimage("", zimage_model)
        emb_n, pooled_emb_n, text_ids_n = compute_text_embeddings_zimage(word, zimage_model)
        
        # Use Gumbel-Softmax to sample from preserved set
        # Get the original embedding shape for this word
        seq_len, hidden_dim = embedding_shape_dict[word]
        
        # Detach preserved_matrix to prevent gradient graph accumulation
        preserved_matrix = preserved_matrix_dict[word].detach()
        
        # matmul: [1, n] @ [n, seq_len*hidden_dim] -> [1, seq_len*hidden_dim]
        emb_r_flat = torch.matmul(
            gumbel_softmax(one_hot_dict[word].bfloat16(), temperature=gumbel_temp, hard=gumbel_hard),
            preserved_matrix,
        )
        
        # Reshape to [1, seq_len, hidden_dim] - keep on GPU 0 for now
        emb_r = emb_r_flat.reshape(1, seq_len, hidden_dim).to(device_0)
        
        # Clear intermediate variables to prevent accumulation
        del preserved_matrix, emb_r_flat
        
        assert emb_r.shape == emb_n.shape, f"Shape mismatch: {emb_r.shape} != {emb_n.shape}"
        
        # Randomly select time step
        tmp_index = np.random.choice([0, 1, 2, 3])
        if tmp_index == 0:
            t_enc_ddpm = torch.tensor([1000.0], device=device_0)
        elif tmp_index == 1:
            t_enc_ddpm = torch.tensor([750.0], device=device_0)
        elif tmp_index == 2:
            t_enc_ddpm = torch.tensor([500.0], device=device_0)
        elif tmp_index == 3:
            t_enc_ddpm = torch.tensor([250.0], device=device_0)
        
        with torch.no_grad():
            # Use original model to generate latents
            z, latent_image_ids = latent_sample(
                model_orig,
                zimage_model.scheduler,
                1,
                num_channels_latents,
                512,
                512,
                emb_n.to(device_0),
                pooled_emb_n.to(device_0),
                text_ids_n.to(device_0),
                timesteps=ddim_steps,
            )
            
            # Clean GPU 0 cache after first sample
            torch.cuda.empty_cache()
            
            z_r, latent_image_ids_r = latent_sample(
                model_orig,
                zimage_model.scheduler,
                1,
                num_channels_latents,
                512,
                512,
                emb_r.to(device_0),
                pooled_emb_n.to(device_0),
                text_ids_n.to(device_0),
                timesteps=ddim_steps,
            )
            
            # Clean GPU 0 cache after second sample
            torch.cuda.empty_cache()
            
            # Get noise prediction from original model
            # Extract device ID from device string (e.g., "cuda:0" -> 0)
            device_0_id = int(devices[0].split(":")[1])
            t_enc_gpu0 = t_enc_ddpm.to(device_0)
            
            e_0_org = predict_noise(
                model_orig,
                z,
                emb_0.to(device_0),
                text_ids_0.to(device_0),
                latent_image_ids,
                pooled_emb_0.to(device_0),
                timesteps=t_enc_gpu0,
                device_id=device_0_id,
            )
            
            e_n_org = predict_noise(
                model_orig,
                z,
                emb_n.to(device_0),
                text_ids_n.to(device_0),
                latent_image_ids,
                pooled_emb_n.to(device_0),
                timesteps=t_enc_gpu0,
                device_id=device_0_id,
            )
            
            e_r_org = predict_noise(
                model_orig,
                z_r,
                emb_r.to(device_0),
                text_ids_n.to(device_0),
                latent_image_ids_r,
                pooled_emb_n.to(device_0),
                timesteps=t_enc_gpu0,
                device_id=device_0_id,
            )
            
            # Clean GPU 0 cache after all predictions
            del t_enc_gpu0
            torch.cuda.empty_cache()
        
        e_0_org.requires_grad = False
        e_n_org.requires_grad = False
        e_r_org.requires_grad = False
        
        # Clean GPU 1 cache before starting training model predictions
        # This is critical to avoid OOM on first iteration
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_1)
        
        # Get noise prediction from training model
        # Extract device ID from device string (e.g., "cuda:1" -> 1)
        device_1_id = int(devices[1].split(":")[1])
        
        # Use Flow Matching inversion for original model predictions
        delta_sigma = 1.0
        z_n_org_pred = delta_sigma * e_n_org
        z_0_org_pred = delta_sigma * e_0_org
        z_r_org_pred = delta_sigma * e_r_org
        
        # Two-stage training - compute predictions only when needed to avoid OOM
        if i % pgd_num_steps == 0:
            # Stage 1: Update model parameters - need both predictions
            # Use gradient accumulation with aggressive memory cleanup
            model.zero_grad()
            total_loss = 0.0
            
            # First: compute e_n_wo_prompt, use it, backward, then free
            z_gpu1 = z.to(device_1, non_blocking=True)
            emb_n_gpu1 = emb_n.to(device_1, non_blocking=True)
            text_ids_n_gpu1 = text_ids_n.to(device_1, non_blocking=True)
            latent_image_ids_gpu1 = latent_image_ids.to(device_1, non_blocking=True)
            pooled_emb_n_gpu1 = pooled_emb_n.to(device_1, non_blocking=True)
            t_enc_gpu1 = t_enc_ddpm.to(device_1, non_blocking=True)
            
            e_n_wo_prompt = predict_noise(
                model,
                z_gpu1,
                emb_n_gpu1,
                text_ids_n_gpu1,
                latent_image_ids_gpu1,
                pooled_emb_n_gpu1,
                timesteps=t_enc_gpu1,
                device_id=device_1_id,
            )
            
            # Immediately compute and use for loss
            z_n_wo_prompt_pred = delta_sigma * e_n_wo_prompt
            del e_n_wo_prompt
            
            # Move target tensors to GPU 1 for first loss
            z_0_target = z_0_org_pred.to(device_1, non_blocking=True)
            z_n_target = z_n_org_pred.to(device_1, non_blocking=True)
            
            loss_n = criteria(
                z_n_wo_prompt_pred,
                z_0_target - (negative_guidance * (z_n_target - z_0_target)),
            )
            
            # Backward first loss immediately to free computation graph
            loss_n = loss_n.float()
            loss_n.backward()
            total_loss += loss_n.item()
            
            # Aggressive cleanup after first backward
            del z_n_wo_prompt_pred, z_0_target, z_n_target, loss_n
            del z_gpu1, emb_n_gpu1, text_ids_n_gpu1, latent_image_ids_gpu1, pooled_emb_n_gpu1, t_enc_gpu1
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_1)
            
            # Second: compute e_r_wo_prompt
            z_r_gpu1 = z_r.to(device_1, non_blocking=True)
            emb_r_gpu1 = emb_r.to(device_1, non_blocking=True)
            latent_image_ids_r_gpu1 = latent_image_ids_r.to(device_1, non_blocking=True)
            t_enc_gpu1 = t_enc_ddpm.to(device_1, non_blocking=True)
            text_ids_n_gpu1 = text_ids_n.to(device_1, non_blocking=True)
            pooled_emb_n_gpu1 = pooled_emb_n.to(device_1, non_blocking=True)
            
            e_r_wo_prompt = predict_noise(
                model,
                z_r_gpu1,
                emb_r_gpu1,
                text_ids_n_gpu1,
                latent_image_ids_r_gpu1,
                pooled_emb_n_gpu1,
                timesteps=t_enc_gpu1,
                device_id=device_1_id,
            )
            
            # Immediately compute and use for loss
            z_r_wo_prompt_pred = delta_sigma * e_r_wo_prompt
            del e_r_wo_prompt
            
            # Move target tensor for second loss
            z_r_target = z_r_org_pred.to(device_1, non_blocking=True)
            
            loss_r = criteria(z_r_wo_prompt_pred, z_r_target)
            
            # Backward second loss
            loss_r = loss_r.float()
            loss_r.backward(retain_graph=False)  # Explicitly release graph
            total_loss += loss_r.item()
            
            # Aggressive cleanup after second backward
            del z_r_wo_prompt_pred, z_r_target, loss_r
            del z_r_gpu1, emb_r_gpu1, latent_image_ids_r_gpu1, t_enc_gpu1, text_ids_n_gpu1, pooled_emb_n_gpu1
            
            # Force immediate cleanup
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_1)
            
            # Update optimizer with accumulated gradients
            losses.append(total_loss)
            pbar.set_postfix({"ESD Loss": total_loss})
            # Don't save to history_dict every step to avoid memory leak
            # history_dict = save_to_dict(total_loss, "loss", history_dict)
            opt.step()
            opt.zero_grad(set_to_none=True)  # Set to None to free memory completely
        else:
            # Stage 2: Update one-hot vector (adversarial optimization) - only need e_r
            model.zero_grad()
            opt.zero_grad()
            
            # Compute e_r_wo_prompt only (Stage 2)
            z_r_gpu1 = z_r.to(device_1, non_blocking=True)
            emb_r_gpu1 = emb_r.to(device_1, non_blocking=True)
            latent_image_ids_r_gpu1 = latent_image_ids_r.to(device_1, non_blocking=True)
            t_enc_gpu1 = t_enc_ddpm.to(device_1, non_blocking=True)
            text_ids_n_gpu1 = text_ids_n.to(device_1, non_blocking=True)
            pooled_emb_n_gpu1 = pooled_emb_n.to(device_1, non_blocking=True)
            
            e_r_wo_prompt = predict_noise(
                model,
                z_r_gpu1,
                emb_r_gpu1,
                text_ids_n_gpu1,
                latent_image_ids_r_gpu1,
                pooled_emb_n_gpu1,
                timesteps=t_enc_gpu1,
                device_id=device_1_id,
            )
            
            # Immediately compute and use
            z_r_wo_prompt_pred = delta_sigma * e_r_wo_prompt
            del e_r_wo_prompt
            
            # Move target tensor to GPU 1
            z_r_target = z_r_org_pred.to(device_1, non_blocking=True)
            
            # Maximize preserved loss (by minimizing negative number)
            loss = -criteria(z_r_wo_prompt_pred, z_r_target).float()
            
            # Clean up
            del z_r_wo_prompt_pred, z_r_target
            del z_r_gpu1, emb_r_gpu1, latent_image_ids_r_gpu1, t_enc_gpu1, text_ids_n_gpu1, pooled_emb_n_gpu1
            
            losses_onehot.append(loss.item())
            pbar.set_postfix({"EAP Loss": loss.item()})
            loss.backward(retain_graph=False)  # Explicitly release graph
            opt_one_hot.step()
            opt_one_hot.zero_grad(set_to_none=True)  # Set to None to free memory
            model.zero_grad(set_to_none=True)  # Set to None to free memory
            
            # Clean up loss
            del loss
            
            # Force immediate cleanup
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_1)
        
        # Clean up ALL tensors from this iteration to prevent accumulation
        del z, z_r, latent_image_ids, latent_image_ids_r
        del e_0_org, e_n_org, e_r_org
        del z_n_org_pred, z_0_org_pred, z_r_org_pred
        del emb_0, emb_n, emb_r, pooled_emb_0, pooled_emb_n, text_ids_0, text_ids_n
        
        # Force aggressive cleanup every iteration
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        if torch.cuda.is_available():
            torch.cuda.synchronize(device_0)
            torch.cuda.synchronize(device_1)
        
        # Save periodically and aggressive cleanup to prevent memory leak
        if (i + 1) % 100 == 0:
            save_history(losses, name, word_print)
            save_history_onehot(losses_onehot, name, word_print)
            
            # Clear history_dict to prevent memory accumulation (only keep initial setup)
            keys_to_keep = ["preserved_set_0", "one_hot_dict_0"]
            history_dict = {k: v for k, v in history_dict.items() if k in keys_to_keep}
            
            # Aggressive cleanup every 100 steps
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device_1)  # Reset memory stats
            if torch.cuda.is_available():
                torch.cuda.synchronize(device_0)
                torch.cuda.synchronize(device_1)
                
            print(f"\n[Memory] Step {i+1}: Forced cleanup completed")
    
    model.eval()
    print("[Training] EAP training finished!")
    
    # Save model
    save_model_lora(model, name, output_dir)
    save_history(losses, name, word_print)
    save_history_onehot(losses_onehot, name, word_print)


def save_model_lora(model, name, output_dir):
    """
    Save Position-Masked LoRA weights
    Format: text_masked_lora.safetensors
    """
    folder_path = f"{output_dir}/{name}"
    os.makedirs(folder_path, exist_ok=True)
    
    # Extract LoRA weights
    lora_state_dict = {}
    for idx, layer in enumerate(model.layers):
        attn = layer.attention
        if hasattr(attn.to_q, 'lora_down'):
            lora_state_dict[f'layers.{idx}.attention.to_q.lora_down.weight'] = attn.to_q.lora_down.weight.data.cpu()
            lora_state_dict[f'layers.{idx}.attention.to_q.lora_up.weight'] = attn.to_q.lora_up.weight.data.cpu()
        if hasattr(attn.to_k, 'lora_down'):
            lora_state_dict[f'layers.{idx}.attention.to_k.lora_down.weight'] = attn.to_k.lora_down.weight.data.cpu()
            lora_state_dict[f'layers.{idx}.attention.to_k.lora_up.weight'] = attn.to_k.lora_up.weight.data.cpu()
    
    save_path = os.path.join(folder_path, "text_masked_lora.safetensors")
    save_file(lora_state_dict, save_path)
    print(f"[Saved] Position-Masked LoRA weights to {save_path}")


def save_history(losses, name, word_print):
    """Save loss history"""
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss.txt", "w") as f:
        f.writelines([f"{l}\n" for l in losses])
    plot_loss(losses, f"{folder_path}/loss.png", word_print, n=3)


def save_history_onehot(losses, name, word_print):
    """Save one-hot loss history"""
    folder_path = f"models/{name}"
    os.makedirs(folder_path, exist_ok=True)
    with open(f"{folder_path}/loss_onehot.txt", "w") as f:
        f.writelines([f"{l}\n" for l in losses])
    plot_loss(losses, f"{folder_path}/loss_onehot.png", word_print, n=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train ZImage EAP",
        description="Train ZImage using EAP method for concept erasure with Position-Masked LoRA"
    )
    parser.add_argument("--prompt", help="Concept to erase", type=str, required=True)
    parser.add_argument(
        "--start_guidance", 
        help="Guidance for start image", 
        type=float, 
        required=False, 
        default=3
    )
    parser.add_argument(
        "--negative_guidance", 
        help="Guidance for negative training", 
        type=float, 
        required=False, 
        default=1
    )
    parser.add_argument(
        "--iterations", 
        help="Number of iterations", 
        type=int, 
        required=False, 
        default=1000
    )
    parser.add_argument(
        "--lr", 
        help="Learning rate", 
        type=float, 
        required=False, 
        default=1e-4
    )
    parser.add_argument(
        "--devices", 
        help="CUDA devices to train on", 
        type=str, 
        required=False, 
        default="0,1"
    )
    parser.add_argument(
        "--output_dir",
        help="Output directory for models",
        type=str,
        required=False,
        default="models"
    )
    parser.add_argument(
        "--seperator",
        help="Separator for multiple concepts",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        "--image_size", 
        help="Image size", 
        type=int, 
        required=False, 
        default=512
    )
    parser.add_argument(
        "--gumbel_lr", 
        help="Learning rate for Gumbel-Softmax", 
        type=float, 
        required=False, 
        default=1e-3
    )
    parser.add_argument(
        "--gumbel_temp", 
        help="Temperature for Gumbel-Softmax", 
        type=float, 
        required=False, 
        default=2
    )
    parser.add_argument(
        "--gumbel_hard",
        help="Hard mode for Gumbel-Softmax (0: soft, 1: hard)",
        type=int,
        required=False,
        default=1,
        choices=[0, 1],
    )
    parser.add_argument(
        "--gumbel_k_closest", 
        help="Number of closest tokens", 
        type=int, 
        required=False, 
        default=1000
    )
    parser.add_argument(
        "--gumbel_num_centers",
        help="Number of K-means centers",
        type=int,
        required=False,
        default=100,
    )
    parser.add_argument(
        "--ddim_steps", 
        help="DDIM steps for sampling", 
        type=int, 
        required=False, 
        default=4
    )
    parser.add_argument(
        "--lora_rank",
        help="LoRA rank",
        type=int,
        required=False,
        default=64
    )
    parser.add_argument(
        "--target_lora_layers",
        help="Target layers for LoRA (comma-separated indices or None for all layers)",
        type=str,
        required=False,
        default=None,
    )
    
    args = parser.parse_args()
    
    # When CUDA_VISIBLE_DEVICES is set, PyTorch remaps devices to cuda:0, cuda:1, etc.
    # So we always use cuda:0 and cuda:1 regardless of the actual physical GPU IDs
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        devices = ["cuda:0", "cuda:1"]
        print(f"[Device Mapping] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}, using {devices}")
    else:
        devices = [f"cuda:{int(d.strip())}" for d in args.devices.split(",")]
    
    train_eap(
        prompt=args.prompt,
        start_guidance=args.start_guidance,
        negative_guidance=args.negative_guidance,
        iterations=args.iterations,
        lr=args.lr,
        devices=devices,
        output_dir=args.output_dir,
        seperator=args.seperator,
        image_size=args.image_size,
        ddim_steps=args.ddim_steps,
        gumbel_k_closest=args.gumbel_k_closest,
        gumbel_num_centers=args.gumbel_num_centers,
        gumbel_lr=args.gumbel_lr,
        gumbel_temp=args.gumbel_temp,
        gumbel_hard=args.gumbel_hard,
        lora_rank=args.lora_rank,
        target_lora_layers=args.target_lora_layers,
    )

