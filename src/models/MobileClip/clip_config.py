"""
Plain Python class for CLIPModel configuration.

This module defines a plain 'ClipConfig' class to hold configuration
and provides a helper function to populate it from the main VLMChatConfig.
"""

from typing import Dict, Any, Optional
from utils.config import VLMChatConfig

class ClipConfig:
    """
    A plain Python class (POPO) for CLIP-specific configuration.
    
    This object is instantiated and populated by the get_clip_config function.
    """
    def __init__(self):
        self.model_name: str = "MobileCLIP2-S0"
        self.pretrained_model_path: str = "/home/patrick/mobileclip2_s0.pt"
        self.model_kwargs: Dict[str, Any] = {}

# --- Helper Function ---

def get_clip_config(config: VLMChatConfig) -> ClipConfig:
    """
    Parses and returns a validated ClipConfig object.
    
    This function creates a ClipConfig instance and populates it
    by extracting parameters from the main VLMChatConfig.
    """
    
    cfg = ClipConfig()
    
    # --- Define defaults (from the old Pydantic model) ---
    default_model_name = "MobileCLIP2-S0"
    default_pretrained_path = "./mobileclip2_s0.pt"
    default_model_kwargs = {}

    # --- Extract values from VLMChatConfig (or use defaults) ---
    
    # Use getattr to safely get values, falling back to the default
    cfg.model_name = getattr(config.model, "clip_model_name", default_model_name)
    cfg.pretrained_model_path = getattr(config.model, "clip_pretrained_path", default_pretrained_path)
    
    # --- Apply specific logic from your script ---
    
    # Check if user provided kwargs in the config
    user_kwargs = getattr(config.model, "clip_model_kwargs", None)
    if user_kwargs is not None:
        cfg.model_kwargs.update(user_kwargs)
    
    # Apply the image_mean/std logic IF not already set by the user
    # This logic is specific to the MobileCLIP models in your script.
    if "MobileCLIP" in cfg.model_name:
        if 'image_mean' not in cfg.model_kwargs and 'image_std' not in cfg.model_kwargs:
            if not (cfg.model_name.endswith("S3") or cfg.model_name.endswith("S4") or cfg.model_name.endswith("L-14")):
                cfg.model_kwargs["image_mean"] = (0, 0, 0)
                cfg.model_kwargs["image_std"] = (1, 1, 1)

    return cfg