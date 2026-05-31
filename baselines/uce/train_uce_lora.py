#!/usr/bin/env python
# coding: utf-8
"""
@date: 2025.12.30
UCE (Unified Concept Editing) for ZImage with Position-Masked LoRA
Adapts UCE closed-form algorithm to ZImage architecture
"""

import torch
torch.set_grad_enabled(False)
import argparse
import os
import copy
import time
import gc
from safetensors.torch import save_file
from diffusers import ZImagePipeline


def compute_text_embeddings_zimage(pipe, prompts, max_sequence_length=256, device="cuda"):
    """
    Compute text embeddings using ZImage's Qwen tokenizer and text encoder
    
    Returns:
        embeddings: dict mapping prompt -> (last_token_embedding, pooled_embedding)
    """
    embeddings = {}
    
    for prompt in prompts:
        if prompt in embeddings:
            continue
        
        # Tokenize with Qwen tokenizer
        text_inputs = pipe.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_tensors="pt",
        )
        
        text_input_ids = text_inputs.input_ids.to(device)
        attention_mask = text_inputs.attention_mask.to(device)
        
        # Get text encoder outputs
        outputs = pipe.text_encoder(
            input_ids=text_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Extract hidden states - ZImage uses hidden_states[-2] (second to last)
        if hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
            # Use -2 layer (as ZImage pipeline does)
            prompt_embeds = outputs.hidden_states[-2]
        elif hasattr(outputs, 'last_hidden_state'):
            prompt_embeds = outputs.last_hidden_state
        else:
            raise ValueError("Cannot extract hidden states from text encoder")
        
        # Extract only valid tokens (mask out padding)
        valid_tokens = prompt_embeds[0][attention_mask[0].bool()]
        
        # Use last valid token (most representative)
        if len(valid_tokens) > 0:
            last_token_embed = valid_tokens[-1]  # Last valid token
        else:
            last_token_embed = prompt_embeds[0, 0, :]  # Fallback to first token
        
        # Compute pooled embedding (mean of valid tokens)
        if len(valid_tokens) > 0:
            pooled_embed = valid_tokens.mean(dim=0)
        else:
            pooled_embed = prompt_embeds[0, 0, :]
        
        embeddings[prompt] = (last_token_embed, pooled_embed)
    
    return embeddings


def UCE_ZImage(
    model_id,
    edit_concepts,
    guide_concepts,
    preserve_concepts,
    erase_scale,
    preserve_scale,
    lamb,
    save_dir,
    exp_name,
    torch_dtype,
    device,
    max_sequence_length,
    lora_rank=64,
    target_layers=None,
):
    """
    Apply UCE algorithm to ZImage transformer attention layers
    
    UCE modifies weights via closed-form solution:
        W_new = mat1 @ mat2^(-1)
    where:
        mat1 = λW + Σ(scale * v* @ c^T)
        mat2 = λI + Σ(scale * c @ c^T)
    
    For ZImage, we apply this to transformer.layers[i].attention.to_q and to_k,
    then decompose ΔW via SVD into low-rank LoRA format.
    """
    print("\n" + "="*60)
    print("UCE for ZImage - Position-Masked LoRA")
    print("="*60)
    
    # Load ZImage pipeline (minimal components)
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"\n[1/5] Loading ZImage model: {model_id}")
    pipe = ZImagePipeline.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir
    )
    pipe = pipe.to(device)
    
    # Extract transformer modules (attention.to_q, attention.to_k)
    transformer = pipe.transformer
    
    # Determine target layers (default: all layers)
    if target_layers is None:
        target_layers = list(range(len(transformer.layers)))
    
    print(f"  Target layers: {target_layers}")
    
    # Collect modules to edit
    uce_modules = []
    uce_module_info = []  # (layer_idx, module_name, module)
    
    for idx in target_layers:
        layer = transformer.layers[idx]
        attn = layer.attention
        
        # to_q and to_k are the attention weight matrices we'll edit
        uce_modules.append(attn.to_q.to(device))
        uce_module_info.append((idx, 'to_q', attn.to_q))
        
        uce_modules.append(attn.to_k.to(device))
        uce_module_info.append((idx, 'to_k', attn.to_k))
    
    # Deep copy original modules for computing guide outputs
    original_modules = copy.deepcopy(uce_modules)
    
    # Free memory by removing heavy components we don't need
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    # Reload minimal pipeline (only text encoder + tokenizer)
    print("\n[2/5] Reloading for text embedding extraction...")
    pipe = ZImagePipeline.from_pretrained(
        model_id,
        vae=None,
        transformer=None,
        torch_dtype=torch_dtype,
        cache_dir=cache_dir
    )
    pipe = pipe.to(device)
    
    # Compute text embeddings for all concepts
    start_time = time.time()
    all_prompts = edit_concepts + guide_concepts + preserve_concepts
    print(f"\n[3/5] Computing text embeddings for {len(all_prompts)} prompts...")
    print(f"  Edit concepts: {edit_concepts}")
    print(f"  Guide concepts: {guide_concepts}")
    print(f"  Preserve concepts: {preserve_concepts}")
    
    text_embeddings = compute_text_embeddings_zimage(
        pipe,
        all_prompts,
        max_sequence_length=max_sequence_length,
        device=device
    )
    
    # Free text encoder
    del pipe
    torch.cuda.empty_cache()
    gc.collect()
    
    # Compute guide outputs (forward through original weights)
    print("\n[4/5] Computing guide outputs with original weights...")
    guide_outputs = {}
    
    # Get transformer hidden_dim from first module
    transformer_hidden_dim = original_modules[0].in_features
    text_embed_dim = text_embeddings[list(text_embeddings.keys())[0]][0].shape[0]
    
    print(f"  Text embedding dim: {text_embed_dim}")
    print(f"  Transformer hidden dim: {transformer_hidden_dim}")
    
    # For UCE, we need a proper projection from text encoder to transformer
    # Linear layer weight shape: [out_features, in_features] = [transformer_hidden_dim, text_embed_dim]
    if text_embed_dim != transformer_hidden_dim:
        print(f"  Creating projection layer: {text_embed_dim} -> {transformer_hidden_dim}")
        text_projection = torch.nn.Linear(text_embed_dim, transformer_hidden_dim, bias=False).to(device, dtype=torch_dtype)
        # Initialize as identity-like: first text_embed_dim rows are identity, rest are zeros
        # Weight matrix shape: [transformer_hidden_dim, text_embed_dim] = [3840, 2560]
        with torch.no_grad():
            if transformer_hidden_dim > text_embed_dim:
                # First text_embed_dim rows: identity matrix [2560, 2560]
                identity_part = torch.eye(text_embed_dim, device=device, dtype=torch_dtype)  # [2560, 2560]
                # Remaining rows: zeros [1280, 2560]
                zero_padding = torch.zeros(transformer_hidden_dim - text_embed_dim, text_embed_dim,
                                          device=device, dtype=torch_dtype)  # [1280, 2560]
                # Stack vertically: [3840, 2560]
                text_projection.weight.data = torch.cat([identity_part, zero_padding], dim=0)
            else:
                # Truncate if needed (shouldn't happen for ZImage)
                identity_part = torch.eye(text_embed_dim, device=device, dtype=torch_dtype)
                text_projection.weight.data = identity_part[:transformer_hidden_dim, :]
        text_projection.eval()
    else:
        text_projection = None
    
    for concept in guide_concepts + preserve_concepts:
        if concept in guide_outputs:
            continue
        
        last_token_embed, pooled_embed = text_embeddings[concept]
        guide_outputs[concept] = []
        
        # Use last_token_embed (more representative)
        emb_to_use = last_token_embed.to(dtype=torch_dtype)
        
        # Project to transformer hidden_dim if needed
        if text_projection is not None:
            emb_to_use = text_projection(emb_to_use)
        
        # Forward through each original module: v = W @ c
        for module in original_modules:
            output = module(emb_to_use)
            guide_outputs[concept].append(output)
    
    # UCE Algorithm: Apply closed-form weight update
    print("\n[5/5] Applying UCE closed-form solution...")
    print(f"  Erase scale: {erase_scale}, Preserve scale: {preserve_scale}, Lambda: {lamb}")
    
    lora_state_dict = {}
    
    for module_idx, (layer_idx, module_name, original_module) in enumerate(uce_module_info):
        W_old = original_modules[module_idx].weight.data
        
        # Initialize UCE matrices
        mat1 = lamb * W_old
        mat2 = lamb * torch.eye(W_old.shape[1], device=device, dtype=torch_dtype)
        
        in_features = W_old.shape[1]
        
        # Helper function to get projected embedding
        def get_projected_embedding(concept_name):
            last_token_embed, pooled_embed = text_embeddings[concept_name]
            emb = last_token_embed.to(dtype=torch_dtype)
            if text_projection is not None:
                emb = text_projection(emb)
            return emb
        
        # Erase concepts
        for erase_concept, guide_concept in zip(edit_concepts, guide_concepts):
            c_i = get_projected_embedding(erase_concept)
            c_i = c_i.unsqueeze(1)  # [D, 1]
            v_i_star = guide_outputs[guide_concept][module_idx].unsqueeze(1)  # [D_out, 1]
            
            mat1 += erase_scale * (v_i_star @ c_i.T)  # [D_out, D]
            mat2 += erase_scale * (c_i @ c_i.T)  # [D, D]
        
        # Preserve concepts
        for preserve_concept in preserve_concepts:
            c_i = get_projected_embedding(preserve_concept)
            c_i = c_i.unsqueeze(1)
            v_i_star = guide_outputs[preserve_concept][module_idx].unsqueeze(1)
            
            mat1 += preserve_scale * (v_i_star @ c_i.T)
            mat2 += preserve_scale * (c_i @ c_i.T)
        
        # Compute new weight via closed-form solution
        W_new = mat1 @ torch.inverse(mat2.float()).to(torch_dtype)
        
        # Compute weight difference
        delta_W = W_new - W_old  # [out_features, in_features]
        
        # Decompose ΔW into low-rank format via SVD: ΔW ≈ U @ S @ V^T
        # We'll keep top-k singular values where k = lora_rank
        try:
            U, S, Vt = torch.linalg.svd(delta_W.float(), full_matrices=False)
            
            # Keep top lora_rank components
            k = min(lora_rank, len(S))
            U_k = U[:, :k]  # [out_features, k]
            S_k = S[:k]  # [k]
            Vt_k = Vt[:k, :]  # [k, in_features]
            
            # LoRA decomposition: ΔW ≈ (U_k @ sqrt(S_k)) @ (sqrt(S_k) @ Vt_k)
            # lora_up: [out_features, rank]
            # lora_down: [rank, in_features]
            sqrt_S_k = torch.sqrt(S_k)
            lora_up = U_k * sqrt_S_k.unsqueeze(0)  # [out_features, k]
            lora_down = (sqrt_S_k.unsqueeze(1) * Vt_k)  # [k, in_features]
            
            # Transpose to match nn.Linear weight format: [out, in]
            lora_down = lora_down  # Already [k, in_features]
            lora_up = lora_up  # Already [out_features, k]
            
            # Save as position-masked LoRA format
            key_prefix = f"layers.{layer_idx}.attention.{module_name}"
            lora_state_dict[f"{key_prefix}.lora_down.weight"] = lora_down.to(torch.float32).contiguous()
            lora_state_dict[f"{key_prefix}.lora_up.weight"] = lora_up.to(torch.float32).contiguous()
            
            # Compute reconstruction error
            delta_W_reconstructed = lora_up @ lora_down
            error = torch.norm(delta_W.float() - delta_W_reconstructed) / torch.norm(delta_W.float())
            
            print(f"  Layer {layer_idx}.{module_name}: SVD rank={k}, reconstruction error={error:.4f}")
        
        except Exception as e:
            print(f"  Warning: SVD failed for layer {layer_idx}.{module_name}: {e}")
            # Fallback: use zero initialization
            out_features, in_features = W_old.shape
            lora_down = torch.zeros(lora_rank, in_features, dtype=torch.float32, device=device)
            lora_up = torch.zeros(out_features, lora_rank, dtype=torch.float32, device=device)
            
            key_prefix = f"layers.{layer_idx}.attention.{module_name}"
            lora_state_dict[f"{key_prefix}.lora_down.weight"] = lora_down.contiguous()
            lora_state_dict[f"{key_prefix}.lora_up.weight"] = lora_up.contiguous()
    
    # Save LoRA weights
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, exp_name + '.safetensors')
    save_file(lora_state_dict, save_path)
    
    end_time = time.time()
    
    print("\n" + "="*60)
    print(f"  UCE completed successfully!")
    print(f"  Time elapsed: {end_time - start_time:.2f}s")
    print(f"  Saved to: {save_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='TrainUCE_ZImage',
        description='UCE for erasing concepts in ZImage with Position-Masked LoRA'
    )
    
    # Concept arguments
    parser.add_argument('--edit_concepts', help='Prompts for concepts to erase (separated by ;)', type=str, required=True)
    parser.add_argument('--guide_concepts', help='Concepts to guide erased concepts towards (separated by ;)', type=str, default=None)
    parser.add_argument('--preserve_concepts', help='Concepts to preserve (separated by ;)', type=str, default=None)
    parser.add_argument('--concept_type', help='Type of concept being erased', choices=['art', 'object', 'style', 'nudity'], type=str, default='object')
    
    # Model arguments
    parser.add_argument('--model_id', help='ZImage model to apply UCE on', type=str, default="Tongyi-MAI/Z-Image-Turbo")
    parser.add_argument('--device', help='CUDA device', type=str, default='cuda:0')
    
    # UCE hyperparameters
    parser.add_argument('--erase_scale', help='Scale for erasing concepts', type=float, default=10.0)
    parser.add_argument('--preserve_scale', help='Scale for preserving concepts', type=float, default=1.0)
    parser.add_argument('--lamb', help='Lambda regularization term', type=float, default=0.1)
    
    # LoRA configuration
    parser.add_argument('--lora_rank', help='LoRA rank for decomposition', type=int, default=64)
    parser.add_argument('--target_layers', help='Target layers (e.g., "0,1,2" or "all")', type=str, default='all')
    
    # Prompt expansion
    parser.add_argument('--expand_prompts', help='Expand prompts with templates', choices=['true', 'false'], type=str, default='false')
    
    # Output arguments
    parser.add_argument('--save_dir', help='Directory to save UCE LoRA weights', type=str, default='uce_models')
    parser.add_argument('--exp_name', help='Experiment name for saved file', type=str, default=None)
    
    args = parser.parse_args()
    
    # Configuration
    device = args.device
    torch_dtype = torch.bfloat16  # ZImage uses bfloat16
    model_id = args.model_id
    
    preserve_scale = args.preserve_scale
    erase_scale = args.erase_scale
    lamb = args.lamb
    lora_rank = args.lora_rank
    
    concept_type = args.concept_type
    expand_prompts = args.expand_prompts
    
    save_dir = args.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    exp_name = args.exp_name
    if exp_name is None:
        exp_name = 'text_masked_lora'
    
    max_sequence_length = 256  # ZImage-Turbo uses 256
    
    # Parse concepts
    edit_concepts = [c.strip() for c in args.edit_concepts.split(';')]
    
    guide_concepts = args.guide_concepts
    if guide_concepts is None:
        # Default guide concepts based on type
        if concept_type == 'art' or concept_type == 'style':
            guide_concepts = 'art'
        elif concept_type == 'nudity':
            guide_concepts = 'person'
        else:
            guide_concepts = 'object'
    guide_concepts = [c.strip() for c in guide_concepts.split(';')]
    
    # Repeat guide concepts to match edit concepts if needed
    if len(guide_concepts) == 1:
        guide_concepts = guide_concepts * len(edit_concepts)
    
    if len(guide_concepts) != len(edit_concepts):
        raise ValueError(f"Length mismatch: {len(edit_concepts)} edit concepts vs {len(guide_concepts)} guide concepts")
    
    # Parse preserve concepts
    if args.preserve_concepts is None:
        preserve_concepts = []
    else:
        preserve_concepts = [c.strip() for c in args.preserve_concepts.split(';')]
    
    # Parse target layers
    if args.target_layers == 'all':
        target_layers = None  # Will use all layers
    else:
        target_layers = [int(x.strip()) for x in args.target_layers.split(',')]
    
    # Expand prompts if requested
    if expand_prompts == 'true':
        edit_concepts_ = copy.deepcopy(edit_concepts)
        guide_concepts_ = copy.deepcopy(guide_concepts)
        
        for concept, guide_concept in zip(edit_concepts_, guide_concepts_):
            if concept_type in ['art', 'style']:
                edit_concepts.extend([
                    f'painting by {concept}',
                    f'art by {concept}',
                    f'artwork by {concept}',
                    f'picture by {concept}',
                    f'style of {concept}'
                ])
                guide_concepts.extend([
                    f'painting by {guide_concept}',
                    f'art by {guide_concept}',
                    f'artwork by {guide_concept}',
                    f'picture by {guide_concept}',
                    f'style of {guide_concept}'
                ])
            else:
                edit_concepts.extend([
                    f'image of {concept}',
                    f'photo of {concept}',
                    f'portrait of {concept}',
                    f'picture of {concept}',
                    f'painting of {concept}'
                ])
                guide_concepts.extend([
                    f'image of {guide_concept}',
                    f'photo of {guide_concept}',
                    f'portrait of {guide_concept}',
                    f'picture of {guide_concept}',
                    f'painting of {guide_concept}'
                ])
    
    print("\n" + "="*60)
    print("UCE Configuration:")
    print(f"  Erasing: {edit_concepts}")
    print(f"  Guiding to: {guide_concepts}")
    print(f"  Preserving: {preserve_concepts}")
    print("="*60)
    
    # Run UCE
    UCE_ZImage(
        model_id,
        edit_concepts,
        guide_concepts,
        preserve_concepts,
        erase_scale,
        preserve_scale,
        lamb,
        save_dir,
        exp_name,
        torch_dtype,
        device,
        max_sequence_length,
        lora_rank=lora_rank,
        target_layers=target_layers,
    )

