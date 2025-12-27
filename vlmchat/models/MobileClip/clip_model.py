"""
CLIP model wrapper with separate image and text encoders.

This module provides:
- CLIPModel: The main orchestrator that uses separate image and text models
             to compute similarity matches between images and text prompts.
"""

import logging
import torch
from typing import List, Optional, Tuple
from PIL import Image

from utils.config import VLMChatConfig
from metrics.metrics_collector import Collector, null_collector
from .clip_image_model import ClipImageModel
from .clip_text_model import ClipTextModel

logger = logging.getLogger(__name__)


class CLIPModel:
    """
    Orchestrator for CLIP image-text matching.
    
    Uses separate ClipImageModel and ClipTextModel to encode images and text,
    then computes cosine similarity between them.
    """
    
    def __init__(self, config: VLMChatConfig, collector: Optional[Collector] = null_collector()):
        self.config = config
        self.collector = collector
        
        # Initialize separate image and text models
        self.image_model = ClipImageModel(config, collector)
        self.text_model = ClipTextModel(config, collector)
    
    @property
    def is_available(self) -> bool:
        """Returns True if both image and text models are available."""
        return (self.image_model._runtime is not None and 
                self.image_model._runtime.is_available and 
                self.text_model._runtime is not None and
                self.text_model._runtime.is_available)
    
    def pre_cache_text_prompts(self, prompts: List[str]) -> None:
        """
        Pre-encodes a list of text prompts and caches the features.
        This is useful for the semantic clusterer.
        """
        self.text_model.pre_cache_prompts(prompts)
    
    def get_matches(self, 
                    image: Image.Image, 
                    prompts: List[str]) -> List[Tuple[float, str]]:
        """
        Compares an image against a list of text prompts and returns
        a sorted list of (confidence, text) tuples.
        
        Args:
            image: PIL Image to compare
            prompts: List of text prompts to match against
            
        Returns:
            A list of (score, text) tuples, sorted by score descending.
        """
        if not self.is_available:
            logger.error("CLIP models not available")
            return []
        
        try:
            # 1. Encode the image
            image_features = self.image_model.encode(image)
            
            # 2. Encode the text (with caching support)
            text_features = self.text_model.encode(prompts, use_cache=True)
            
            # 3. Calculate similarity
            with torch.no_grad(), torch.cuda.amp.autocast():
                # Cosine similarity = dot product of normalized features
                # Multiply by 100 as per original logic
                raw_similarity = (100.0 * image_features @ text_features.T).cpu().numpy().flatten()
            
            # 4. Format results
            results = list(zip(raw_similarity, prompts))
            
            # Sort by confidence, descending
            results.sort(key=lambda x: x[0], reverse=True)
            
            return [(float(score), str(text)) for score, text in results]

        except Exception as e:
            logger.error(f"Error during CLIP matching: {e}", exc_info=True)
            return []


# Example usage:
if __name__ == "__main__":
    
    # --- Imports for __main__ ---
    import os
    import traceback

    # Check if dependencies available
    try:
        import open_clip
        OPEN_CLIP_AVAILABLE = True
    except ImportError:
        OPEN_CLIP_AVAILABLE = False

    if not OPEN_CLIP_AVAILABLE:
        print("Cannot run example: 'open_clip' or 'mobileclip' library not installed.")
    else:
        print("--- CLIPModel __main__ Example ---")
        
        # --- Setup: Load Config (adjust path as needed) ---
        # This assumes a VLMChatConfig file exists.
        # For a simple test, we can create a mock config or trust defaults.
        # Let's create a minimal config object.
        class MockConfig:
            def __init__(self):
                self.model = self # Mock the nested structure
                self.clip_model_name = "MobileCLIP2-S0"
                self.clip_pretrained_path = "models/MobileClip/mobileclip2_s0.pt"
                # Add model_kwargs based on logic
                self.clip_model_kwargs = {}
                if not (self.clip_model_name.endswith("S3") or self.clip_model_name.endswith("S4") or self.clip_model_name.endswith("L-14")):
                    self.clip_model_kwargs = {"image_mean": (0, 0, 0), "image_std": (1, 1, 1)}

        mock_config = MockConfig()
        
        # --- Image and Text Setup ---
        image_path = "models/MobileClip/trail-riders.jpg"
        text_prompts = ["a horse", "a person riding a horse", "a man wearing a hat"]
        
        if not os.path.exists(image_path):
            print(f"Error: Example image not found: {image_path}")
            print("Please download 'trail-riders.jpg' to run this example.")
        else:
            try:
                # 1. Initialize Model
                print("Initializing CLIPModel...")
                # We pass the mock_config as the VLMChatConfig
                clip_model = CLIPModel(config=mock_config) # type: ignore
                
                if not clip_model.is_available:
                    raise RuntimeError("Failed to initialize CLIP models.")
                
                # 2. Load Image
                image = Image.open(image_path)
                
                # 3. Get Matches
                print(f"Matching image against: {text_prompts}")
                matches = clip_model.get_matches(image, text_prompts)
                
                # 4. Print Results
                print("\n--- Results (Score, Text) ---")
                for score, text in matches:
                    print(f"({score:.2f}, '{text}')")

            except Exception as e:
                print(f"\nAn error occurred during the example: {e}")
                traceback.print_exc()