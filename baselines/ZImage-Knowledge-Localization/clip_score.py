"""
CLIP score calculation for image-text similarity evaluation
"""
import torch


def get_clip_score(clip_model, clip_processor, text, image):
    """
    Calculate CLIP similarity score between text and image
    
    Args:
        clip_model: CLIP model
        clip_processor: CLIP processor
        text: Text description
        image: PIL Image
    
    Returns:
        CLIP similarity score (float)
    """
    # Process inputs
    inputs = clip_processor(
        text=[text],
        images=image,
        return_tensors="pt",
        padding=True
    ).to(clip_model.device)
    
    # Get features
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        clip_score = logits_per_image.item()
    
    return clip_score
