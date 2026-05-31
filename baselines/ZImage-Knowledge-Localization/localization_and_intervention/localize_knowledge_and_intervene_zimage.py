#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Localize Knowledge and Intervene for ZImage Model
Main script for knowledge localization and intervention evaluation on ZImage
"""
import argparse
import json
import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
from functools import partial
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from utils import find_substring_token_indices, get_worker_list_chunk, print_arguments, latents_to_images
from attention_processor import ZImageAttnContCalculatorProcessor, ZImageEmbeddingModifierAttnProcessor
from clip_score import get_clip_score
from dataset import get_knowledge_dataset_class_and_get_list_fn, get_eval_text_for_knowledge
from custom_zimage_pipeline import load_custom_zimage_pipeline


def localize_dominant_blocks(args, pipe, dataset):
    """
    Localize dominant transformer blocks for target knowledge
    
    Args:
        args: Command-line arguments
        pipe: CustomZImagePipeline
        dataset: Knowledge dataset
    
    Returns:
        List of dominant block indices
    """
    print(f"Starting localization for: {dataset.knowledge}")
    
    # Initialize aggregated attention contribution tensor
    # ZImage has 30 single-stream transformer blocks
    num_blocks = len(pipe.transformer.layers)
    aggergated_attn_cont = torch.zeros(num_blocks)
    
    for prompt in tqdm(dataset, desc="Localizing Dominant Blocks"):
        # Find token indices of the knowledge in the prompt
        token_indices = find_substring_token_indices(
            prompt, 
            dataset.knowledge, 
            pipe.tokenizer, 
            "zimage"
        )
        
        if not token_indices:
            print(f"Warning: Could not find '{dataset.knowledge}' tokens in prompt '{prompt}', skipping...")
            continue
        
        # Set attention contribution calculator processors
        for idx, layer in enumerate(pipe.transformer.layers):
            processor = ZImageAttnContCalculatorProcessor(
                token_indices_for_attn_cont_calc=token_indices,
                image_seq_len=1024  # ZImage uses 1024 image tokens (32x32 latent)
            )
            layer.attention.set_processor(processor)
        
        # Generate with the prompt to collect attention contributions
        with torch.no_grad():
            _ = pipe(
                prompt=prompt,
                height=512,
                width=512,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=0.0,  # ZImage-Turbo uses guidance_scale=0
                output_type="latent",
                generator=torch.Generator().manual_seed(args.generator_seed),
            )
        
        # Aggregate attention contributions from all blocks
        for idx, layer in enumerate(pipe.transformer.layers):
            processor = layer.attention.processor
            if processor.attn_contribution_update_count > 0:
                avg_cont = processor.attn_contribution / processor.attn_contribution_update_count
                aggergated_attn_cont[idx] += avg_cont
    
    # Average over all prompts
    aggergated_attn_cont = aggergated_attn_cont / len(dataset)
    
    # Save attention contribution
    save_path = f"{args.results_path}/{dataset.knowledge}/attn_cont.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(aggergated_attn_cont, save_path)
    print(f"Saved attention contribution to {save_path}")
    
    # Get top-k dominant blocks
    top_dominant_blocks = aggergated_attn_cont.topk(args.disable_k_dominant_blocks).indices.tolist()
    
    return top_dominant_blocks

# def evaluate(args, pipe, clip_score_fn, dataset, top_dominant_blocks):
def evaluate(args, pipe, dataset, top_dominant_blocks):
    """
    Evaluate intervention by generating images with modified embeddings
    
    Args:
        args: Command-line arguments
        pipe: CustomZImagePipeline
        clip_score_fn: Function to calculate CLIP score
        dataset: Knowledge dataset
        top_dominant_blocks: List of dominant block indices to intervene
    """
    print(f"Starting evaluation for: {dataset.knowledge}")
    
    res = {}
    
    for prompt in tqdm(dataset, desc="Evaluating"):
        for i in range(args.num_images_per_eval_prompt):
            seed = args.generator_seed + i
            
            try:
                # Generate with intervention
                with torch.no_grad():
                    output = pipe(
                        prompt=prompt,
                        clean_prompt=dataset.get_clean_prompt(prompt),
                        modifier_indices=top_dominant_blocks,
                        height=512,
                        width=512,
                        num_inference_steps=args.num_inference_steps,
                        guidance_scale=0.0,
                        generator=torch.Generator().manual_seed(seed),
                    )
                    
                    image = output.images[0]
                
                # Save image
                file_name = f"{prompt}_{seed}.png"
                save_path = f"{args.results_path}/{dataset.knowledge}/{file_name}"
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                image.save(save_path)
                
                # Calculate CLIP score
                # eval_text = get_eval_text_for_knowledge(args.knowledge_type, dataset.knowledge)
                # clip_score = clip_score_fn(eval_text, image)
                # res[file_name] = clip_score
                
            except Exception as e:
                print(f"Error generating image for '{prompt}' with seed {seed}: {e}")
                continue
    
    # Save results
    results_file = f"{args.results_path}/{dataset.knowledge}/results.json"
    with open(results_file, "w") as f:
        json.dump(res, f, indent=4)
    
    print(f"Saved evaluation results to {results_file}")
    
    # Print summary statistics
    if res:
        avg_score = sum(res.values()) / len(res)
        print(f"Average CLIP score: {avg_score:.4f}")


def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="Localize and intervene knowledge in ZImage model"
    )
    
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=9,
        help="Number of denoising steps (default: 9 for ZImage-Turbo)"
    )
    
    parser.add_argument(
        "--generator_seed",
        type=int,
        default=0,
        help="Random seed for generation"
    )
    
    parser.add_argument(
        "--disable_k_dominant_blocks",
        type=int,
        default=6,
        help="Number of top dominant blocks to disable/modify"
    )
    
    parser.add_argument(
        "--results_path",
        type=str,
        required=True,
        help="Path to save results"
    )
    
    parser.add_argument(
        "--num_images_per_eval_prompt",
        type=int,
        default=3,
        help="Number of images to generate per evaluation prompt"
    )
    
    parser.add_argument(
        "--num_workers",
        type=int,
        default=20,
        help="Total number of parallel workers"
    )
    
    parser.add_argument(
        "--worker_idx",
        type=int,
        required=True,
        help="Index of current worker (0-based)"
    )
    
    parser.add_argument(
        "--knowledge_type",
        type=str,
        required=True,
        choices=["style", "place", "copyright", "animal", "celebrity", "safety"],
        help="Type of knowledge to localize"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="Tongyi-MAI/Z-Image-Turbo",
        help="Path or HuggingFace repo of ZImage model"
    )
    
    args = parser.parse_args()
    return args


def main():
    """Main function"""
    args = parse_args()
    
    print_arguments(args)
    
    # Get dataset class and knowledge list function
    dataset_class, get_knowledge_list_fn = get_knowledge_dataset_class_and_get_list_fn(
        args.knowledge_type, 
        for_model="zimage"
    )
    
    # Split knowledge list among workers
    worker_knowledge_list = get_worker_list_chunk(
        get_knowledge_list_fn(), 
        args.num_workers, 
        args.worker_idx
    )
    
    if not worker_knowledge_list:
        print(f"No knowledge items assigned to worker {args.worker_idx}")
        return
    
    # Create results directory
    if not os.path.exists(args.results_path):
        print(f"Creating results directory: {args.results_path}")
        os.makedirs(args.results_path, exist_ok=True)
    
    # Load ZImage pipeline
    print(f"Loading ZImage model from {args.model_path}...")
    pipe = load_custom_zimage_pipeline(args.model_path)
    print("Model loaded successfully")
    
    # Load CLIP model for evaluation
    # print("Loading CLIP model...")
    # clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True).to("cuda")
    # clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", local_files_only=True)
    # print("CLIP model loaded successfully")
    
    # Create CLIP score function
    # clip_score_fn = partial(get_clip_score, clip_model, clip_processor)
    
    # Process each knowledge item
    for knowledge in worker_knowledge_list:
        print(f"Processing {args.knowledge_type.title()}: {knowledge}")
        
        # Skip if results already exist
        results_file = f"{args.results_path}/{knowledge}/results.json"
        if os.path.exists(results_file):
            print(f"Results already exist for '{knowledge}'. Skipping...")
            continue
        
        # Create knowledge directory
        knowledge_dir = f"{args.results_path}/{knowledge}"
        if not os.path.exists(knowledge_dir):
            os.makedirs(knowledge_dir)
    
        # Step 1: Localize dominant blocks
        print(f"\n[1/2] Localizing dominant blocks...")
        train_dataset = dataset_class(knowledge, "train")
        top_dominant_blocks = localize_dominant_blocks(args, pipe, train_dataset)
        
        print(f"Top {args.disable_k_dominant_blocks} dominant blocks for '{knowledge}':")
        print(f"  Indices: {top_dominant_blocks}")
        
        # Step 2: Evaluate with intervention
        print(f"\n[2/2] Evaluating intervention...")
        eval_dataset = dataset_class(knowledge, "both")
        # evaluate(args, pipe, clip_score_fn, eval_dataset, top_dominant_blocks)
        evaluate(args, pipe, eval_dataset, top_dominant_blocks)


if __name__ == '__main__':
    main()
