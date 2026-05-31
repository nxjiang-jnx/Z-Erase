# coding: UTF-8
"""
    @date:  2024.12.04
    @func:  attention map 
"""
import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import torch
import numpy as np
from diffusers import ZImagePipeline

cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
os.makedirs(cache_dir, exist_ok=True)

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pipe = ZImagePipeline.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    cache_dir=cache_dir,
)
pipe = pipe.to(DEVICE)


def search_sequence_numpy(arr, seq):
    if arr.ndim > 1:
        arr = arr.flatten()
    if seq.ndim > 1:
        seq = seq.flatten()
    
    seq = seq[seq != 0]
    
    if len(seq) == 0:
        return []
    
    result_indices = []
    
    for i in range(len(arr) - len(seq) + 1):
        if np.array_equal(arr[i:i+len(seq)], seq):
            result_indices.extend(range(i, i+len(seq)))
    
    return result_indices

def get_word_index(prompt, attn_words, tokenizer):
    prompt_text_ids = tokenizer(
        prompt,
        padding=False,
        truncation=False,
        return_tensors="np"
    ).input_ids
    
    prompt_tokens = prompt_text_ids.flatten()
    
    word_start = prompt.lower().find(attn_words.lower())
    if word_start == -1:
        return []
    
    for i in range(len(prompt_tokens)):
        for j in range(i+1, min(i+5, len(prompt_tokens)+1)):
            decoded = tokenizer.decode(prompt_tokens[i:j]).strip()
            if decoded.lower() == attn_words.lower():
                result_indices = list(range(i, j))
                return result_indices
    
    return []


if __name__ == "__main__":
    tokenizer_qwen = pipe.tokenizer
    answer = get_word_index("a nude girl with beautiful hair and her breast open to see", "breast", tokenizer_qwen)
    print(answer)
