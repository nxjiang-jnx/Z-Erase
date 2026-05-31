"""
Utility functions for ZImage knowledge localization
"""
import torch
from transformers import PreTrainedTokenizer


def find_substring_token_indices(prompt: str, substr: str, tokenizer: PreTrainedTokenizer, model="zimage"):
    """
    Find token indices of a substring within a prompt for ZImage tokenizer
    
    Args:
        prompt: Full prompt text
        substr: Substring to find (e.g., artist name, place name)
        tokenizer: ZImage tokenizer
        model: Model name (default "zimage")
    
    Returns:
        List of token indices where substr appears
    """
    assert model == "zimage", f"Model {model} not supported"
    
    # Apply chat template to prompt
    messages = [{"role": "user", "content": prompt}]
    prompt_formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    
    # Tokenize full prompt
    prompt_tokens = tokenizer(prompt_formatted).input_ids
    
    # Tokenize substring - try different approaches due to tokenizer quirks
    # First try with space prefix (common in middle of sentence)
    substr_with_space = " " + substr
    substr_tokens_with_space = tokenizer(substr_with_space, add_special_tokens=False).input_ids
    
    # Also try without space
    substr_tokens_no_space = tokenizer(substr, add_special_tokens=False).input_ids
    
    # Try to find either version
    start_idx = -1
    used_tokens = None
    
    # Try with space first
    for i in range(len(prompt_tokens) - len(substr_tokens_with_space) + 1):
        if prompt_tokens[i:i+len(substr_tokens_with_space)] == substr_tokens_with_space:
            start_idx = i
            used_tokens = substr_tokens_with_space
            break
    
    # If not found, try without space
    if start_idx == -1:
        for i in range(len(prompt_tokens) - len(substr_tokens_no_space) + 1):
            if prompt_tokens[i:i+len(substr_tokens_no_space)] == substr_tokens_no_space:
                start_idx = i
                used_tokens = substr_tokens_no_space
                break
    
    if start_idx == -1:
        print(f"Warning: Could not find '{substr}' in prompt '{prompt}'")
        print(f"Prompt tokens: {prompt_tokens}")
        print(f"Substr tokens (with space): {substr_tokens_with_space}")
        print(f"Substr tokens (no space): {substr_tokens_no_space}")
        # Return empty list as fallback
        return []
    
    token_indices = list(range(start_idx, start_idx + len(used_tokens)))
    
    # Verify decoding
    decoded = tokenizer.decode([prompt_tokens[idx] for idx in token_indices])
    if substr not in decoded and decoded.strip() != substr:
        print("============================ Warning ============================")
        print(f"[Warning] Decoded text doesn't match substr exactly")
        print(f"[Warning] Decoded: '{decoded}'")
        print(f"[Warning] Expected: '{substr}'")
        print("=================================================================")
    
    return token_indices


def latents_to_images(pipe, latents):
    """
    Decode latents to images using VAE
    
    Args:
        pipe: ZImage pipeline
        latents: Latent tensors [B, C, H, W]
    
    Returns:
        List of PIL images
    """
    latents = latents.to(pipe.vae.dtype)
    with torch.no_grad():
        images = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
    images = pipe.image_processor.postprocess(images, output_type="pil")
    
    return images


def get_worker_list_chunk(arr, num_workers, worker_idx, print_log=True):
    """
    Split array into chunks for parallel processing
    
    Args:
        arr: List to split
        num_workers: Total number of workers
        worker_idx: Current worker index (0-based)
        print_log: Whether to print chunk info
    
    Returns:
        Chunk of list for this worker
    """
    arr_len = len(arr)
    
    chunk_size = (arr_len + num_workers - 1) // num_workers
    
    start_index = chunk_size * worker_idx
    end_index = min((worker_idx + 1) * chunk_size, arr_len)
    
    if print_log:
        print(f"Choosing chunk ({start_index}:{end_index})")
        if start_index < arr_len:
            print(f"First item of the chunk: \"{arr[start_index]}\"")
            print(f"Last item of the chunk: \"{arr[end_index-1]}\"")
    
    return arr[start_index:end_index]


def print_arguments(args):
    """Print all command-line arguments in formatted way"""
    print("===================== Arguments =====================")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("=====================================================")


def get_text_length_from_embeddings(prompt_embeds_list):
    """
    Get the actual text token length from ZImage prompt embeddings
    
    Args:
        prompt_embeds_list: List of text embeddings (one per sample in batch)
    
    Returns:
        Maximum text length in the batch
    """
    if isinstance(prompt_embeds_list, list):
        return max(emb.shape[0] for emb in prompt_embeds_list)
    else:
        return prompt_embeds_list.shape[1]
