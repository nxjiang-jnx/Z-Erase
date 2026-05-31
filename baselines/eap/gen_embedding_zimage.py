"""
@date: 2025.12.28
@func: ZImage version of embedding generation and token search
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import random
import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import ZImagePipeline
from torch.autograd import Variable
from tqdm import tqdm
from utils_zimage import gumbel_softmax, save_to_dict

LEN_EN_3K_VOCAB = 3000


def get_english_tokens():
    """Get English 3000 vocabulary"""
    data_path = "data/english_3000.csv"
    df = pd.read_csv(data_path)
    vocab = {}
    for ir, row in df.iterrows():
        vocab[row["word"]] = ir
    assert len(vocab) == LEN_EN_3K_VOCAB
    return vocab


def detect_special_tokens(text):
    """Detect if the text contains special characters"""
    text = text.lower()
    for i in range(len(text)):
        if text[i] not in "abcdefghijklmnopqrstuvwxyz</>":
            return True
    return False


def retrieve_embedding_token(model_name, query_token, vocab="EN3K"):
    """Retrieve the embedding of a token from the saved embedding matrix"""
    if vocab == "EN3K":
        if model_name == "ZImage-Turbo":
            embedding_matrix = torch.load("models/embedding_matrix_dict_EN3K_zimage.pt")
        else:
            raise ValueError("model_name should be 'ZImage-Turbo'")
        if query_token in embedding_matrix:
            return embedding_matrix[query_token]
    else:
        raise ValueError("vocab should be 'EN3K'")


@torch.no_grad()
def create_embedding_matrix(
    pipe, 
    start=0, 
    end=1000, 
    model_name="ZImage-Turbo", 
    save_mode="array", 
    vocab="EN3K"
):
    """
    Create embedding matrix
    Generate text embeddings for all vocab tokens
    """
    tokenizer_vocab = get_vocab(pipe, vocab=vocab)
    
    if save_mode == "array":
        all_embeddings = []
        for token in tqdm(tokenizer_vocab.keys()):
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            
            # Use ZImage's tokenizer and text encoder
            text_inputs = pipe.tokenizer(
                [token],
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
            
            all_embeddings.append(prompt_embeds)
        
        return torch.cat(all_embeddings, dim=0)
    
    elif save_mode == "dict":
        all_embeddings = {}
        for token in tqdm(tokenizer_vocab.keys()):
            if tokenizer_vocab[token] < start or tokenizer_vocab[token] >= end:
                continue
            
            # Use ZImage's tokenizer and text encoder
            text_inputs = pipe.tokenizer(
                [token],
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
            
            all_embeddings[token] = prompt_embeds
        
        return all_embeddings
    else:
        raise ValueError("save_mode should be either 'array' or 'dict'")


@torch.no_grad()
def search_closest_tokens(
    concept,
    pipe,
    k=10,
    reshape=True,
    sim="cosine",
    model_name="ZImage-Turbo",
    ignore_special_tokens=True,
    vocab="EN3K",
):
    """
    Given a concept (e.g., "nudity"), search for the top-k tokens in the embedding space that are closest to it
    """
    tokenizer_vocab = get_vocab(pipe, vocab=vocab)
    # Reverse dictionary for indexing
    tokenizer_vocab_indexing = {v: k for k, v in tokenizer_vocab.items()}
    
    # Get the embedding of the central concept
    text_inputs = pipe.tokenizer(
        [concept],
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
        central_concept_embeds = outputs.last_hidden_state
    elif hasattr(outputs, 'hidden_states') and len(outputs.hidden_states) > 0:
        central_concept_embeds = outputs.hidden_states[-1]
    else:
        raise ValueError("Cannot extract hidden states from text encoder")
    
    # Calculate the similarity between the central concept and all tokens
    all_similarities = []
    if vocab == "EN3K":
        if model_name == "ZImage-Turbo":
            embedding_matrix = torch.load("models/embedding_matrix_array_EN3K_zimage.pt")
        else:
            raise ValueError("model_name should be 'ZImage-Turbo'")
        
        # Use the embedding of the first token (usually the most important)
        central_concept_embeds = central_concept_embeds[:, 0, :]
        embedding_matrix = embedding_matrix[:, 0, :].to(central_concept_embeds.device)
        
        if reshape:
            central_concept_embeds = central_concept_embeds.view(central_concept_embeds.size(0), -1)
            embedding_matrix = embedding_matrix.view(embedding_matrix.size(0), -1)
        
        if sim == "cosine":
            similarities = F.cosine_similarity(central_concept_embeds, embedding_matrix, dim=-1)
        elif sim == "l2":
            similarities = -F.pairwise_distance(central_concept_embeds, embedding_matrix, p=2)
        
        all_similarities.append(similarities)
    else:
        raise ValueError("vocab should be 'EN3K'")
    
    similarities = torch.cat(all_similarities, dim=0)
    # Sort the similarities
    sorted_similarities, indices = torch.sort(similarities, descending=True)
    print(f"sorted_similarities: {sorted_similarities[:10]}")
    print(f"indices: {indices[:10]}")
    
    sim_dict = {}
    for im, i in enumerate(indices):
        if ignore_special_tokens:
            if detect_special_tokens(tokenizer_vocab_indexing[i.item()]):
                continue
        token = tokenizer_vocab_indexing[i.item()]
        sim_dict[token] = sorted_similarities[im]
    
    top_k_tokens = list(sim_dict.keys())[:k]
    return top_k_tokens, sim_dict


def save_embedding_matrix(pipe, model_name="ZImage-Turbo", save_mode="array", vocab="EN3K"):
    """Save embedding matrix to file"""
    if vocab == "EN3K":
        embedding_matrix = create_embedding_matrix(
            pipe, 
            start=0, 
            end=LEN_EN_3K_VOCAB, 
            model_name=model_name, 
            save_mode=save_mode, 
            vocab="EN3K"
        )
        if model_name == "ZImage-Turbo":
            print("[ZImage-Turbo] Saving embedding matrix")
            os.makedirs("models", exist_ok=True)
            torch.save(embedding_matrix, f"models/embedding_matrix_{save_mode}_EN3K_zimage.pt")
    else:
        raise ValueError("vocab should be 'EN3K'")


@torch.no_grad()
def get_vocab(pipe, vocab="EN3K"):
    """Get vocabulary"""
    if vocab == "EN3K":
        tokenizer_vocab = get_english_tokens()  # vocab_size 3,000
    else:
        # If needed, use the full vocabulary of ZImage
        if pipe is not None:
            tokenizer_vocab = pipe.tokenizer.vocab
    
    return tokenizer_vocab


def my_kmean(sorted_sim_dict, num_centers, compute_mode="numpy"):
    """K-means clustering"""
    if compute_mode == "numpy":
        import numpy as np
        from sklearn.cluster import KMeans
        
        similarities = np.array([sorted_sim_dict[token].item() for token in sorted_sim_dict])
        similarities = similarities.reshape(-1, 1)
        kmeans = KMeans(n_clusters=num_centers, random_state=0).fit(similarities)
        cluster_centers = kmeans.cluster_centers_
    elif compute_mode == "torch":
        from torch_kmeans import KMeans
        
        similarities = torch.stack([sorted_sim_dict[token] for token in sorted_sim_dict])
        similarities = torch.unsqueeze(similarities, dim=0)
        similarities = torch.unsqueeze(similarities, dim=2)  # [1, N, 1]
        kmeans = KMeans(n_clusters=num_centers).fit(similarities)
        cluster_centers = kmeans.cluster_centers
    
    # Find the token closest to each cluster center
    cluster_dict = {}
    for i, center in enumerate(cluster_centers):
        closest_token = None
        closest_similarity = -float("inf")
        for j, token in enumerate(sorted_sim_dict):
            similarity = sorted_sim_dict[token].item()
            if abs(similarity - center) < abs(closest_similarity - center):
                closest_similarity = similarity
                closest_token = token
        cluster_dict[closest_token] = (closest_token, closest_similarity, i)
    
    return cluster_dict


@torch.no_grad()
def learn_k_means_from_input_embedding(sim_dict, num_centers=5, compute_mode="numpy"):
    """
    Use k-means clustering on the input embedding
    """
    if num_centers <= 0:
        print("Number of centers should be greater than 0. Returning the tokens themselves.")
        return list(sim_dict.keys())
    if len(list(sim_dict.keys())) <= num_centers:
        print("Number of tokens is less than the number of centers. Returning the tokens themselves.")
        return list(sim_dict.keys())
    
    return list(my_kmean(sim_dict, num_centers, compute_mode).keys())


def create_prompt(word, retrieve=True, vocab="EN3K"):
    """Create prompt embedding"""
    if retrieve:
        return retrieve_embedding_token(model_name="ZImage-Turbo", query_token=word, vocab=vocab)


if __name__ == "__main__":
    # Test code
    pipe = ZImagePipeline.from_pretrained(
        "Tongyi-MAI/Z-Image-Turbo", 
        torch_dtype=torch.bfloat16
    ).to("cuda")
    
    # Generate embedding matrix
    if not os.path.exists("models/embedding_matrix_dict_EN3K_zimage.pt"):
        save_embedding_matrix(pipe, model_name="ZImage-Turbo", save_mode="dict", vocab="EN3K")
    
    if not os.path.exists("models/embedding_matrix_array_EN3K_zimage.pt"):
        save_embedding_matrix(pipe, model_name="ZImage-Turbo", save_mode="array", vocab="EN3K")
    
    # Test search
    erased_words = ["nude"]
    gumbel_k_closest = 1000
    gumbel_num_centers = 50
    
    tokens_embedding = []
    all_sim_dict = dict()
    for word in erased_words:
        top_k_tokens, sorted_sim_dict = search_closest_tokens(word, pipe, k=gumbel_k_closest)
        tokens_embedding.extend(top_k_tokens)
        all_sim_dict[word] = {key: sorted_sim_dict[key] for key in top_k_tokens}
    
    print(f"Found {len(all_sim_dict[erased_words[0]])} similar tokens for '{erased_words[0]}'")

