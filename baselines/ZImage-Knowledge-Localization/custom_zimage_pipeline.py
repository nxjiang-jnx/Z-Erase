"""
Custom ZImage Pipeline with support for embedding modification
Extends the base ZImagePipeline to support knowledge localization and intervention
"""
import sys
from pathlib import Path

# Add diffusers to path
diffusers_path = Path(__file__).resolve().parent.parent / "diffusers" / "src"
sys.path.insert(0, str(diffusers_path))

import torch
from typing import List, Optional, Union
from diffusers import ZImagePipeline
from attention_processor import (
    ZImageCachingAttnProcessor,
    ZImageAttnContCalculatorProcessor,
    ZImageEmbeddingModifierAttnProcessor
)


class CustomZImagePipeline(ZImagePipeline):
    """
    Extended ZImage Pipeline with custom attention processors for knowledge localization
    """
    
    def set_caching_processors(self):
        """Set caching processors to all transformer blocks for attention analysis"""
        for idx, layer in enumerate(self.transformer.layers):
            processor = ZImageCachingAttnProcessor(idx, image_seq_len=1024)
            layer.attention.set_processor(processor)
    
    def set_attn_cont_calculator_processors(self, token_indices):
        """Set attention contribution calculator processors"""
        for idx, layer in enumerate(self.transformer.layers):
            processor = ZImageAttnContCalculatorProcessor(
                token_indices_for_attn_cont_calc=token_indices,
                image_seq_len=1024  # ZImage uses 1024 image tokens (32x32 latent)
            )
            layer.attention.set_processor(processor)
    
    def set_embedding_modifier_processors(self, modifier_indices: List[int]):
        """
        Set embedding modifier processors to specified blocks
        
        Args:
            modifier_indices: List of block indices to modify
        """
        for idx, layer in enumerate(self.transformer.layers):
            if idx in modifier_indices:
                processor = ZImageEmbeddingModifierAttnProcessor(
                    mode=ZImageEmbeddingModifierAttnProcessor.ProcessorMode.REPLACE_TEXT_HIDDEN_STATES,
                    image_seq_len=1024
                )
            else:
                processor = ZImageEmbeddingModifierAttnProcessor(
                    mode=ZImageEmbeddingModifierAttnProcessor.ProcessorMode.NONE,
                    image_seq_len=1024
                )
            layer.attention.set_processor(processor)
    
    def clear_attention_caches(self):
        """Clear all cached attention maps"""
        for layer in self.transformer.layers:
            if hasattr(layer.attention.processor, 'clear_maps'):
                layer.attention.processor.clear_maps()
            elif hasattr(layer.attention.processor, 'clear_cache'):
                layer.attention.processor.clear_cache()
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        clean_prompt: Optional[Union[str, List[str]]] = None,
        modifier_indices: Optional[List[int]] = None,
        **kwargs
    ):
        """
        Extended generation with optional clean prompt intervention
        
        Args:
            prompt: Original prompt(s) with knowledge
            clean_prompt: Clean prompt(s) without knowledge (for intervention)
            modifier_indices: Indices of blocks to modify with clean embeddings
            **kwargs: Other arguments passed to parent __call__
        
        Returns:
            Generated images
        """
        # If intervention is requested, do two-pass generation
        if clean_prompt is not None and modifier_indices is not None and len(modifier_indices) > 0:
            return self._generate_with_intervention(
                prompt, clean_prompt, modifier_indices, **kwargs
            )
        else:
            # Normal generation without intervention
            return super().__call__(prompt=prompt, **kwargs)
    
    def _generate_with_intervention(
        self,
        prompt: Union[str, List[str]],
        clean_prompt: Union[str, List[str]],
        modifier_indices: List[int],
        **kwargs
    ):
        """
        Generate with embedding modification intervention
        
        Two-pass approach:
        1. First pass: encode clean prompt and save text embeddings
        2. Second pass: generate with original prompt but replace embeddings in target blocks
        """
        # Ensure prompts are lists
        if isinstance(prompt, str):
            prompt = [prompt]
        if isinstance(clean_prompt, str):
            clean_prompt = [clean_prompt]
        
        # Pass 1: Set processors to SAVE mode and encode clean prompt
        for idx, layer in enumerate(self.transformer.layers):
            if idx in modifier_indices:
                processor = ZImageEmbeddingModifierAttnProcessor(
                    mode=ZImageEmbeddingModifierAttnProcessor.ProcessorMode.SAVE_TEXT_HIDDEN_STATES,
                    image_seq_len=1024
                )
            else:
                processor = ZImageEmbeddingModifierAttnProcessor(
                    mode=ZImageEmbeddingModifierAttnProcessor.ProcessorMode.NONE,
                    image_seq_len=1024
                )
            layer.attention.set_processor(processor)
        
        # Encode clean prompt to save embeddings
        clean_embeds = self.encode_prompt(
            prompt=clean_prompt,
            device=self._execution_device,
            max_sequence_length=kwargs.get('max_sequence_length', 512)
        )
        
        # Do a dummy forward pass to save clean embeddings (use minimal steps)
        _ = super().__call__(
            prompt=clean_prompt,
            num_inference_steps=1,
            output_type="latent",
            **{k: v for k, v in kwargs.items() if k not in ['num_inference_steps', 'output_type']}
        )
        
        # Pass 2: Set processors to REPLACE mode and generate with original prompt
        for idx, layer in enumerate(self.transformer.layers):
            if idx in modifier_indices:
                processor = ZImageEmbeddingModifierAttnProcessor(
                    mode=ZImageEmbeddingModifierAttnProcessor.ProcessorMode.REPLACE_TEXT_HIDDEN_STATES,
                    image_seq_len=1024
                )
                # Transfer saved embeddings from previous processor
                old_processor = layer.attention.processor
                if hasattr(old_processor, 'saved_text_hidden_states'):
                    processor.saved_text_hidden_states = old_processor.saved_text_hidden_states
            else:
                processor = ZImageEmbeddingModifierAttnProcessor(
                    mode=ZImageEmbeddingModifierAttnProcessor.ProcessorMode.NONE,
                    image_seq_len=1024
                )
            layer.attention.set_processor(processor)
        
        # Generate final images with original prompt but modified embeddings
        output = super().__call__(prompt=prompt, **kwargs)
        
        return output


def load_custom_zimage_pipeline(model_name_or_path="Tongyi-MAI/Z-Image-Turbo", **kwargs):
    """
    Load custom ZImage pipeline with knowledge localization support
    
    Args:
        model_name_or_path: Model path or HuggingFace repo
        **kwargs: Additional arguments for from_pretrained
    
    Returns:
        CustomZImagePipeline instance
    """
    import os
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    os.makedirs(cache_dir, exist_ok=True)
    
    pipe = CustomZImagePipeline.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir,
        **kwargs
    )
    pipe = pipe.to("cuda")
    
    return pipe
