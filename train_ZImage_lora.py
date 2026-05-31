#!/usr/bin/env python
# coding=utf-8
"""
    @date:  2025.12.5-12.22
    @func:  Training ZImage LoRA to erase concept
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import argparse
import yaml
import time
import copy
import itertools
import logging
import math
import random
import json
import shutil
from contextlib import nullcontext
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torch.utils.checkpoint
import transformers

from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from PIL.ImageOps import exif_transpose
from safetensors.torch import save_file
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms.functional import crop
from tqdm.auto import tqdm
from transformers import PretrainedConfig

import diffusers
from diffusers import AutoencoderKL, ZImagePipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params, free_memory
from diffusers.utils import check_min_version, is_wandb_available

from lora_dataset import LoraDataset, collate_data_fn
# Use gradient-preserving loss functions for training
from utils.calc_loss_gradfix import (
    calculate_ca_loss_with_grad as calculate_ca_loss,
    calculate_esd_loss_with_grad as calculate_esd_loss,
)
from utils.calc_loss import (
    calculate_erase_loss,
    calculate_lial_loss,
    calculate_preserve_loss,
    calculate_attn_loss,
)
from utils.training_monitor import TrainingMonitor
from utils.custom_scheduler import CustomFlowMatchEulerDiscreteScheduler
from utils.lagrangian import EU
from utils.zimage_text_lora import (
    apply_text_masked_lora_to_transformer,
    collect_attention_maps,
    clear_attention_maps,
    calculate_text_masked_attn_loss,
    ZImageTextMaskedAttnProcessor,
)

print("All modules imported successfully")

if is_wandb_available():
    import wandb

check_min_version("0.32.0.dev0")
logger = logging.getLogger(__name__)


def load_zimage_components(args):
    
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load ZImage pipeline to extract text encoder and tokenizer
    from diffusers import ZImagePipeline
    
    pipe = ZImagePipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    )
    
    text_encoder = pipe.text_encoder
    tokenizer = pipe.tokenizer
    transformer = pipe.transformer
    vae = pipe.vae
    scheduler = pipe.scheduler
    
    return text_encoder, tokenizer, transformer, vae, scheduler, pipe


def finetune(args):
    logging_dir = Path(args.output_dir, args.logging_dir)
    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
    
    devices = [f'cuda:{int(d.strip())}' for d in args.devices.split(',')]
    print("Devices:", devices)
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    transformers.utils.logging.set_verbosity_error()
    diffusers.utils.logging.set_verbosity_error()

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)

    model_id = Path(args.output_dir).name
    
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load all components from pipeline in one go (avoids duplicate loading)
    text_encoder, tokenizer, transformer, vae, scheduler_base, pipe = load_zimage_components(args)
    
    text_encoders = [text_encoder]
    tokenizers = [tokenizer]
    
    # Use CustomFlowMatchEulerDiscreteScheduler wrapper
    noise_scheduler = CustomFlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
        cache_dir=cache_dir
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    
    # Clean up pipeline after extracting components
    del pipe
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Move transformer to device
    transformer = transformer.to(devices[0])
    
    # Freeze all non-trainable parameters
    transformer.requires_grad_(False)
    vae.requires_grad_(False)
    for encoder in text_encoders:
        if encoder is not None:
            encoder.requires_grad_(False)

    # Set mixed precision dtype
    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    vae.to(transformer.device, dtype=weight_dtype)
    transformer.to(transformer.device, dtype=weight_dtype)
    for encoder in text_encoders:
        if encoder is not None:
            encoder.to(transformer.device, dtype=weight_dtype)

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()
    
    use_text_masked_lora = getattr(args, 'use_text_masked_lora', True)
    
    if use_text_masked_lora:
        print("[TEXT-MASKED LORA] Applying Position-Masked LoRA")
        
        lora_rank = getattr(args, 'rank', 64)
        target_layers = getattr(args, 'target_lora_layers', None)
        
        trainable_params = apply_text_masked_lora_to_transformer(
            transformer,
            target_layers=target_layers,  # None = all layers
            lora_rank=lora_rank,
            image_seq_len=1024,  # Fixed for ZImage
        )
        
        transformer_lora_parameters = trainable_params
        
        # Print trainable parameters
        trainable_count = sum(p.numel() for p in trainable_params)
        all_params = sum(p.numel() for p in transformer.parameters())
        print(f"\n[Position-Masked LoRA] Parameters:")
        print(f"  - Total transformer params: {all_params:,}")
        print(f"  - Trainable (text-only) params: {trainable_count:,}")
        print(f"  - Trainable ratio: {100 * trainable_count / all_params:.6f}%\n")
        
    else:    
        if args.lora_layers is not None:
            target_modules = [layer.strip() for layer in args.lora_layers.split(",")]
        else:
            target_modules = []
            
            # layers = [19, 22, 21, 20, 17, 16]
            
            # unified layers
            for i in range(30):
                target_modules.extend([
                    # FFN/MLP layers
                    # f"layers.{i}.feed_forward.w1",   # Input projection to hidden_dim
                    # f"layers.{i}.feed_forward.w2",   # Output projection from hidden_dim
                    # f"layers.{i}.feed_forward.w3",   # Gating projection
                    
                    # Adaptive LayerNorm (global conditioning)
                    # f"layers.{i}.adaLN_modulation.0",
                    # f"layers.{i}.adaLN_modulation.1",
                    
                    # Attention layers
                    # f"layers.{i}.attention.to_q",
                    # f"layers.{i}.attention.to_k",
                    f"layers.{i}.attention.to_v",
                    f"layers.{i}.attention.to_out.0",
                ])
            
            # context_refiner
            for i in range(8):  
                target_modules.extend([
                        f"context_refiner.{i}.attention.to_q",
                        f"context_refiner.{i}.attention.to_k",
                        f"context_refiner.{i}.attention.to_v",
                ])
        
            # Configure LoRA
            transformer_lora_config = LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
                init_lora_weights="gaussian",
                target_modules=target_modules,
            )
            transformer.add_adapter(transformer_lora_config)
        
            # Print trainable parameters
            trainable_params = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
            all_params = sum(p.numel() for p in transformer.parameters())
            print(f"\n[LoRA] Parameters:")
            print(f"  - Total params: {all_params:,}")
            print(f"  - Trainable params: {trainable_params:,}")
            print(f"  - Trainable ratio: {100 * trainable_params / all_params:.4f}%\n")
            
            transformer_lora_parameters = list(filter(lambda p: p.requires_grad, transformer.parameters()))

    # Make sure the trainable params are in float32
    if args.mixed_precision == "fp16":
        for param in transformer_lora_parameters:
            if param.requires_grad:
                param.data = param.data.float()

    freeze_text_encoder = True
    print("[INFO] free text encoder", freeze_text_encoder)
    

    # Optimization parameters
    transformer_parameters_with_lr = {"params": transformer_lora_parameters, "lr": float(args.learning_rate)}
    params_to_optimize = [transformer_parameters_with_lr]

    # Optimizer creation
    if args.optimizer.lower() == "adamw":
        if args.use_8bit_adam:
            try:
                import bitsandbytes as bnb
            except ImportError:
                raise ImportError(
                    "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
                )
            optimizer_class = bnb.optim.AdamW8bit
        else:
            optimizer_class = torch.optim.AdamW
        optimizer = optimizer_class(
            params_to_optimize,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=float(args.adam_weight_decay),
            eps=float(args.adam_epsilon),
        )

    criteria = torch.nn.MSELoss()

    # Dataset and DataLoaders creation
    train_dataset = LoraDataset(
        instance_data_root=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        key_word=args.key_word,
        tokenizer_t5=tokenizer,  # ZImage uses Qwen tokenizer
        size=args.resolution,
        repeats=args.repeats,
        center_crop=args.center_crop,
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate_data_fn(examples, args.with_prior_preservation),
        num_workers=args.dataloader_num_workers,
    )

    # Text embedding computation function for ZImage (using Qwen3Model)
    def compute_text_embeddings(prompt, text_encoders, tokenizers):
        """
        Compute text embeddings using ZImage's Qwen3Model encoder
        Returns format compatible with ZImage transformer
        """
        with torch.no_grad():
            text_encoder = text_encoders[0]
            tokenizer = tokenizers[0]
            
            if isinstance(prompt, str):
                prompt = [prompt]
            
            text_inputs = tokenizer(
                prompt,
                padding="max_length",
                max_length=args.max_sequence_length,
                truncation=True,
                return_tensors="pt",
            )
            
            text_input_ids = text_inputs.input_ids.to(text_encoder.device)
            attention_mask = text_inputs.attention_mask.to(text_encoder.device)
            
            outputs = text_encoder(
                input_ids=text_input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            if hasattr(outputs, 'last_hidden_state'):
                prompt_embeds = outputs.last_hidden_state
            elif hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
                prompt_embeds = outputs.hidden_states[-1]
            else:
                raise ValueError("Cannot extract hidden states from Qwen3Model output")
            
            attention_mask_expanded = attention_mask.unsqueeze(-1).expand(prompt_embeds.shape)
            sum_embeddings = torch.sum(prompt_embeds * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            pooled_prompt_embeds = sum_embeddings / sum_mask
            
            # Text IDs for positional encoding (similar to Flux)
            batch_size = prompt_embeds.shape[0]
            seq_len = prompt_embeds.shape[1]
            text_ids = torch.zeros(batch_size, seq_len, 3).to(device=text_encoder.device, dtype=prompt_embeds.dtype)
            
            return prompt_embeds, pooled_prompt_embeds, text_ids

    # Pre-compute text embeddings if using fixed prompts
    if freeze_text_encoder and not train_dataset.custom_instance_prompts:
        instance_prompt_hidden_states, instance_pooled_prompt_embeds, instance_text_ids = compute_text_embeddings(
            args.instance_prompt, text_encoders, tokenizers
        )

    if not train_dataset.custom_instance_prompts:
        if freeze_text_encoder:
            prompt_embeds = instance_prompt_hidden_states
            pooled_prompt_embeds = instance_pooled_prompt_embeds
            text_ids = instance_text_ids

    vae_config_shift_factor = vae.config.shift_factor
    vae_config_scaling_factor = vae.config.scaling_factor
    vae_config_block_out_channels = vae.config.block_out_channels
    
    if args.cache_latents:
        latents_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(
                    transformer.device, non_blocking=True, dtype=weight_dtype
                )
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)

    # Scheduler and math around the number of training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    # Recalculate total training steps
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    initial_global_step = 0

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
    )
    
    # Initialize training monitor for loss curve plotting
    enable_monitoring = getattr(args, 'enable_monitoring', True)
    training_monitor = TrainingMonitor(
        output_dir=args.output_dir,
        enable_monitoring=enable_monitoring
    )
    
    print("[INFO] negative prompt!", str(args.prompt_b))
    
    # Initialize lagrangian for balancing forget (upper) and retain (lower) tasks
    # lagrangian dynamically adjusts the weight between forget loss and retain loss
    use_lagrangian = getattr(args, 'use_lagrangian', True)
    lagrangian = None
    
    if use_lagrangian:
        device = transformer.device
        lagrangian = EU(
            device=device,
            gamma=getattr(args, 'lagrangian_gamma', 0.01),
            w_lr=getattr(args, 'lagrangian_w_lr', 0.3),
            error=getattr(args, 'lagrangian_error', 0.001),
            log_loss=getattr(args, 'lagrangian_log_loss', False),
        )
        print(f"[lagrangian] Initialized with gamma={lagrangian.error}, w_lr={lagrangian.w_opt.param_groups[0]['lr']}, error={lagrangian.error}")
        print("[INFO] lagrangian method is ENABLED")
    else:
        print("[INFO] lagrangian method is DISABLED")
    
    # Training loop
    for epoch in range(first_epoch, args.num_train_epochs):
        transformer.train()
        vae.eval()
        for encoder in text_encoders:
            if encoder is not None:
                encoder.eval()
        
        for step, batch in enumerate(train_dataloader):
            # Only clear cache periodically if memory issues occur
            # if torch.cuda.is_available() and step > 0 and step % 50 == 0:
            #     torch.cuda.empty_cache()
            
            if use_text_masked_lora:
                clear_attention_maps(transformer)
            
            # Calculate both forget loss (upper) and retain loss (lower)
            esd_loss, t_enc_ddpm = calculate_ca_loss(
                args, batch, compute_text_embeddings, text_encoders, tokenizers, 
                transformer, noise_scheduler_copy, batch["prompts"], vae, criteria, 
                negative_guidance=args.negative_guidance, weight_dtype=weight_dtype, 
                neg_prompts=str(args.prompt_b), start_guidance=0, ddim_steps=9,
                latents_cache=latents_cache if args.cache_latents else None,
                step=step
            )
            
            # Calculate attention loss using position-masked attention maps
            if use_text_masked_lora:
                attn_maps = collect_attention_maps(transformer)
                remove_indices = batch['remove_indices'][0]
                
                if attn_maps is not None and remove_indices is not None and len(remove_indices) > 0:
                    attn_loss = calculate_text_masked_attn_loss(attn_maps, remove_indices)
                else:
                    attn_loss = torch.tensor(0.0, device=transformer.device)
                
                clear_attention_maps(transformer)
            else:
                attn_loss, _ = calculate_attn_loss(
                    args, batch, compute_text_embeddings, text_encoders, tokenizers, 
                    transformer, noise_scheduler_copy, batch["prompts"], vae, criteria, 
                    negative_guidance=args.negative_guidance, weight_dtype=weight_dtype, 
                    neg_prompts=str(args.prompt_b), start_guidance=0, ddim_steps=9,
                    latents_cache=latents_cache if args.cache_latents else None,
                    step=step
                )
            
            # Combine upper losses
            forget_loss = float(args.lamb1) * esd_loss + float(args.lamb2) * attn_loss
            
            
            # (B) lower (preserve): retain task
            preserve_loss, _ = calculate_preserve_loss(
                args, batch, compute_text_embeddings, text_encoders, tokenizers, 
                transformer, noise_scheduler_copy, batch["prompts"], vae, criteria, 
                negative_guidance=args.negative_guidance, weight_dtype=weight_dtype, 
                neg_prompts=str(args.prompt_b), start_guidance=0, ddim_steps=9,
                latents_cache=latents_cache if args.cache_latents else None,
                step=step
            )
            
            # Combine lower losses: preserve
            retain_loss = float(args.lamb3) * preserve_loss
            
            # Use lagrangian to dynamically balance forget and retain losses, or simple sum if disabled
            if use_lagrangian:
                weighted_loss = lagrangian.get_weighted_loss(retain_loss, forget_loss)
            else:
                weighted_loss = forget_loss + retain_loss
            
            # Logging
            logs = {
                "forget_loss": forget_loss.detach().item(),
                "retain_loss": retain_loss.detach().item(),
                "lial": esd_loss.detach().item(),
                "attn": attn_loss.detach().item() if isinstance(attn_loss, torch.Tensor) else attn_loss,
                "preserve": preserve_loss.detach().item(),
                "prompt": batch["prompts"],
                "index": batch['remove_indices'][0],
            }
            if use_lagrangian:
                logs["lagrangian_weight"] = lagrangian.w.detach().item()
            progress_bar.set_postfix(**logs)
            
            # Record metrics for training monitor
            training_monitor.log_step(
                step=global_step,
                esd_loss=esd_loss.detach().item(),
                attn_loss=attn_loss.detach().item() if isinstance(attn_loss, torch.Tensor) else attn_loss,
                preserve_loss=preserve_loss.detach().item(),
                total_loss=weighted_loss.detach().item(),
                lr=lr_scheduler.get_last_lr()[0]
            )
            
            # Backward and optimize
            weighted_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            
            # Save preserve_loss value before deletion for lagrangian update
            preserve_loss_value = preserve_loss.detach().item() if isinstance(preserve_loss, torch.Tensor) else preserve_loss
            
            # del weighted_loss, forget_loss, retain_loss, esd_loss, preserve_loss
            # if isinstance(attn_loss, torch.Tensor):
            #     del attn_loss
            # Only clear cache periodically if memory issues occur
            # if torch.cuda.is_available() and global_step % 50 == 0:
            #     torch.cuda.empty_cache()
            
            # Update lagrangian weights: re-evaluate retain loss after gradient update
            if use_lagrangian:
                lagrangian_update_freq = getattr(args, 'lagrangian_update_freq', 1)
                if global_step % lagrangian_update_freq == 0:
                    retain_loss_updated = float(args.lamb3) * preserve_loss_value
                    lagrangian.update(retain_loss_updated, curr_lr=lr_scheduler.get_last_lr()[0])
            
            progress_bar.update(1)
            global_step += 1
            
            if global_step % args.checkpointing_steps == 0:
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                
                # Save position-masked LoRA weights
                if use_text_masked_lora:
                    lora_state_dict = {}
                    for idx, layer in enumerate(transformer.layers):
                        attn = layer.attention
                        if hasattr(attn.to_q, 'lora_down'):
                            lora_state_dict[f'layers.{idx}.attention.to_q.lora_down.weight'] = attn.to_q.lora_down.weight.data
                            lora_state_dict[f'layers.{idx}.attention.to_q.lora_up.weight'] = attn.to_q.lora_up.weight.data
                        if hasattr(attn.to_k, 'lora_down'):
                            lora_state_dict[f'layers.{idx}.attention.to_k.lora_down.weight'] = attn.to_k.lora_down.weight.data
                            lora_state_dict[f'layers.{idx}.attention.to_k.lora_up.weight'] = attn.to_k.lora_up.weight.data
                    
                    save_file(lora_state_dict, os.path.join(save_path, "text_masked_lora.safetensors"))
                    print(f"\n[Position-Masked LoRA] Saved to {save_path}/text_masked_lora.safetensors")
                else:
                    transformer = transformer.to(weight_dtype)
                    transformer_lora_layers = get_peft_model_state_dict(transformer)
                    ZImagePipeline.save_lora_weights(
                        save_directory=save_path,
                        transformer_lora_layers=transformer_lora_layers,
                    )
                
                logger.info(f"Saved state to {save_path}")
                
                # Generate and save loss curves periodically
                if enable_monitoring:
                    training_monitor.generate_plots()

            if global_step >= args.max_train_steps:
                break
            
    # Save final model
    final_save_path = args.output_dir
    
    if use_text_masked_lora:
        lora_state_dict = {}
        for idx, layer in enumerate(transformer.layers):
            attn = layer.attention
            if hasattr(attn.to_q, 'lora_down'):
                lora_state_dict[f'layers.{idx}.attention.to_q.lora_down.weight'] = attn.to_q.lora_down.weight.data
                lora_state_dict[f'layers.{idx}.attention.to_q.lora_up.weight'] = attn.to_q.lora_up.weight.data
            if hasattr(attn.to_k, 'lora_down'):
                lora_state_dict[f'layers.{idx}.attention.to_k.lora_down.weight'] = attn.to_k.lora_down.weight.data
                lora_state_dict[f'layers.{idx}.attention.to_k.lora_up.weight'] = attn.to_k.lora_up.weight.data
        
        save_file(lora_state_dict, os.path.join(final_save_path, "text_masked_lora.safetensors"))
        print(f"\n[Position-Masked LoRA] Final model saved to {final_save_path}/text_masked_lora.safetensors")
    else:
        transformer = transformer.to(weight_dtype)
        transformer_lora_layers = get_peft_model_state_dict(transformer)
        ZImagePipeline.save_lora_weights(
            save_directory=final_save_path,
            transformer_lora_layers=transformer_lora_layers,
        )
    
    # Finalize training monitor: save metrics and generate plots
    if enable_monitoring:
        training_monitor.finish_training()
    
    print("Training Finished! Congratulations!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ZImage model with LoRA for concept erasure.")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    args = parser.parse_args()
    
    def read_config(config_path):
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    config = read_config(args.config)
    for key, value in config.items():
        setattr(args, key, value)
    finetune(args)