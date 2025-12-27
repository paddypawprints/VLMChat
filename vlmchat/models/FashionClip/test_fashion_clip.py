#!/usr/bin/env python3
"""
Test script for the refactored FashionClip model.

Tests the new BaseModel-based architecture with the same functionality
as the original FashionClip-openclip.py sample.
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from PIL import Image
from models.FashionClip.fashion_clip_model import FashionClipModel
from utils.config import VLMChatConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_fashion_clip():
    """Test the FashionClip model with the hat.jpg example."""
    
    logger.info("=" * 70)
    logger.info("FashionClip Model Test")
    logger.info("=" * 70)
    
    # 1. Create a minimal config
    config = VLMChatConfig()
    
    # 2. Initialize the model
    logger.info("\n--- Initializing FashionClipModel ---")
    model = FashionClipModel(config)
    logger.info(f"Current runtime: {model.current_runtime()}")
    
    # 3. Load test image
    image_path = Path(__file__).parent / "hat.jpg"
    if not image_path.exists():
        logger.error(f"Test image not found: {image_path}")
        logger.info("Please ensure hat.jpg exists in the FashionClip directory")
        return
    
    logger.info(f"\n--- Loading test image: {image_path} ---")
    image = Image.open(image_path)
    logger.info(f"Image size: {image.size}")
    
    # 4. Define fashion prompts
    prompts = ["a hat", "a t-shirt", "shoes"]
    logger.info(f"\n--- Testing with prompts: {prompts} ---")
    
    # 5. Get matches
    logger.info("\n--- Running inference ---")
    matches = model.get_matches(image, prompts)
    
    # 6. Display results
    logger.info("\n--- Results ---")
    logger.info("Label probabilities:")
    for prob, text in matches:
        logger.info(f"  {text:15s}: {prob:.4f} ({prob*100:.2f}%)")
    
    # 7. Test caching
    logger.info("\n--- Testing prompt caching ---")
    model.pre_cache_text_prompts(prompts)
    matches_cached = model.get_matches(image, prompts)
    
    logger.info("Results with cached prompts:")
    for prob, text in matches_cached:
        logger.info(f"  {text:15s}: {prob:.4f} ({prob*100:.2f}%)")
    
    # 8. Test runtime switching (should fail gracefully since only openclip exists)
    logger.info("\n--- Testing runtime info ---")
    logger.info(f"Current runtime: {model.current_runtime()}")
    
    # 9. Test direct encoding methods
    logger.info("\n--- Testing direct encoding methods ---")
    image_features = model.encode_image(image)
    logger.info(f"Image features shape: {image_features.shape}")
    
    text_features = model.encode_text(["a hat"])
    logger.info(f"Text features shape: {text_features.shape}")
    
    logger.info("\n" + "=" * 70)
    logger.info("✅ FashionClip model test completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    try:
        test_fashion_clip()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
