# coding: utf-8
"""
DiT Knowledge Localization (NIPS 2025) adapted for ZImage-Turbo
"""
import os
import torch
from PIL import Image

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ['ORT_DISABLE_THREAD_AFFINITY'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


MODIFIER_LAYERS = [19, 22, 21, 20, 17, 16]
CLEAN_PROMPT = "a cat"

PROMPT = "A realistic photo of three cats"

MODEL_ID = "Tongyi-MAI/Z-Image-Turbo"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 512
NUM_INFERENCE_STEPS = 9
GUIDANCE_SCALE = 0.0
SEED = 362436

OUTPUT_PATH = "./output_DiT_Local_image.png"

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from utils.logger import create_logger

zimage_kl_path = project_root / "ZImage-Knowledge-Localization"
if not zimage_kl_path.exists():
    raise FileNotFoundError(f"ZImage-Knowledge-Localization directory not found at {zimage_kl_path}")
sys.path.insert(0, str(zimage_kl_path))

from custom_zimage_pipeline import CustomZImagePipeline, load_custom_zimage_pipeline


def load_model():
    print(f"[Custom Pipeline] Loading model from {MODEL_ID}")
    pipe = load_custom_zimage_pipeline(
        model_name_or_path=MODEL_ID,
    )
    pipe = pipe.to(DEVICE)
    print(f"✓ Successfully loaded CustomZImagePipeline")
    
    return pipe


def main():
    output_dir = os.path.dirname(OUTPUT_PATH) if os.path.dirname(OUTPUT_PATH) else "."
    os.makedirs(output_dir, exist_ok=True)
    
    logger = create_logger(output_dir)
    
    pipe = load_model()
    
    generator = torch.Generator(device=DEVICE).manual_seed(SEED)
    
    image = pipe(
        prompt=PROMPT,
        clean_prompt=CLEAN_PROMPT,
        modifier_indices=MODIFIER_LAYERS,
        height=IMG_SIZE,
        width=IMG_SIZE,
        num_inference_steps=NUM_INFERENCE_STEPS,
        guidance_scale=GUIDANCE_SCALE,
        generator=generator
    ).images[0]
    
    image.save(OUTPUT_PATH)
    print(f"Saved image to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

