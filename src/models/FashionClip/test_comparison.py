#!/usr/bin/env python3
"""
Comparison test: Original FashionClip sample vs. refactored BaseModel version.

This verifies that the refactored code produces identical results to the original.
"""

import sys
import logging
from pathlib import Path
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image
from models.FashionClip.fashion_clip_model import FashionClipModel
from utils.config import VLMChatConfig

# Setup logging
logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs for cleaner output
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def test_original_style():
    """Run the original sample code style."""
    logger.info("=" * 70)
    logger.info("ORIGINAL STYLE (from FashionClip-openclip.py)")
    logger.info("=" * 70)
    
    import open_clip
    
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms('hf-hub:Marqo/marqo-fashionSigLIP')
    tokenizer = open_clip.get_tokenizer('hf-hub:Marqo/marqo-fashionSigLIP')
    
    image_path = Path(__file__).parent / "hat.jpg"
    image = preprocess_val(Image.open(image_path)).unsqueeze(0)
    text = tokenizer(["a hat", "a t-shirt", "shoes"])
    
    with torch.no_grad(), torch.cuda.amp.autocast():
        image_features = model.encode_image(image, normalize=True)
        text_features = model.encode_text(text, normalize=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
    logger.info("\nLabel probs (original):")
    probs = text_probs[0].tolist()
    prompts = ["a hat", "a t-shirt", "shoes"]
    for prob, text in zip(probs, prompts):
        logger.info(f"  {text:15s}: {prob:.6f}")
    
    return probs


def test_refactored_style():
    """Run the refactored BaseModel style."""
    logger.info("\n" + "=" * 70)
    logger.info("REFACTORED STYLE (BaseModel architecture)")
    logger.info("=" * 70)
    
    config = VLMChatConfig()
    model = FashionClipModel(config)
    
    image_path = Path(__file__).parent / "hat.jpg"
    image = Image.open(image_path)
    prompts = ["a hat", "a t-shirt", "shoes"]
    
    matches = model.get_matches(image, prompts)
    
    logger.info("\nLabel probs (refactored):")
    # matches are sorted, so we need to reorder to match original order
    probs_dict = {text: prob for prob, text in matches}
    probs = [probs_dict[p] for p in prompts]
    
    for prob, text in zip(probs, prompts):
        logger.info(f"  {text:15s}: {prob:.6f}")
    
    return probs


def compare_results(original_probs, refactored_probs, tolerance=1e-5):
    """Compare results and check if they're within tolerance."""
    logger.info("\n" + "=" * 70)
    logger.info("COMPARISON")
    logger.info("=" * 70)
    
    all_match = True
    prompts = ["a hat", "a t-shirt", "shoes"]
    
    logger.info("\nDifferences:")
    for i, prompt in enumerate(prompts):
        diff = abs(original_probs[i] - refactored_probs[i])
        status = "✅" if diff < tolerance else "❌"
        logger.info(f"  {prompt:15s}: {diff:.2e} {status}")
        if diff >= tolerance:
            all_match = False
    
    logger.info("\n" + "=" * 70)
    if all_match:
        logger.info("✅ SUCCESS: Results match within tolerance!")
    else:
        logger.error("❌ FAILURE: Results differ beyond tolerance!")
    logger.info("=" * 70)
    
    return all_match


if __name__ == "__main__":
    try:
        original_probs = test_original_style()
        refactored_probs = test_refactored_style()
        success = compare_results(original_probs, refactored_probs)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
