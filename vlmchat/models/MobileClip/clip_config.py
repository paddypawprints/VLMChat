"""
Plain Python class for CLIPModel configuration.

This module defines a plain 'ClipConfig' class to hold configuration
and provides a helper function to populate it from the main VLMChatConfig.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import os
from utils.config import VLMChatConfig

def resolve_clip_model_path(model_name: str, pretrained_path: Optional[str] = None) -> str:
    """
    Resolve the CLIP model path by checking multiple locations.
    
    Search order:
    1. Provided pretrained_path (if exists)
    2. HuggingFace cache (~/.cache/huggingface/hub/)
    3. Local ml-mobileclip directory
    4. Return pretrained_path as-is (let OpenCLIP handle it)
    
    Args:
        model_name: Name of the CLIP model (e.g., "MobileCLIP2-S0")
        pretrained_path: Optional path from config
        
    Returns:
        Resolved path to model file
    """
    # If pretrained_path provided and exists, use it
    if pretrained_path:
        p = Path(pretrained_path).expanduser()
        if p.exists():
            return str(p)
    
    # Try HuggingFace cache
    if "MobileCLIP" in model_name:
        hf_cache_base = Path.home() / ".cache" / "huggingface" / "hub"
        
        # Map model names to HF repo names
        repo_map = {
            "MobileCLIP2-S0": "models--apple--MobileCLIP2-S0",
            "MobileCLIP2-S1": "models--apple--MobileCLIP2-S1", 
            "MobileCLIP2-S2": "models--apple--MobileCLIP2-S2",
        }
        
        repo_name = repo_map.get(model_name)
        if repo_name:
            repo_path = hf_cache_base / repo_name
            if repo_path.exists():
                # Find the snapshot directory (most recent)
                snapshots = repo_path / "snapshots"
                if snapshots.exists():
                    snapshot_dirs = [d for d in snapshots.iterdir() if d.is_dir()]
                    if snapshot_dirs:
                        # Use the first snapshot (could sort by mtime for most recent)
                        snapshot_dir = sorted(snapshot_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
                        model_file = snapshot_dir / f"{model_name.lower()}.pt"
                        if model_file.exists():
                            print(f"Found CLIP model in HF cache: {model_file}")
                            return str(model_file)
    
    # Try local ml-mobileclip directory (relative to this file)
    local_path = Path(__file__).parent / "ml-mobileclip" / f"{model_name.lower()}.pt"
    if local_path.exists():
        return str(local_path)
    
    # Fall back to pretrained_path or default
    return pretrained_path or f"./{model_name.lower()}.pt"

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
    configured_path = getattr(config.model, "clip_pretrained_path", default_pretrained_path)
    
    print(f"DEBUG get_clip_config: model_name={cfg.model_name}, configured_path={configured_path}")
    
    # Resolve the actual model path (checks HF cache, local paths, etc.)
    cfg.pretrained_model_path = resolve_clip_model_path(cfg.model_name, configured_path)
    
    print(f"DEBUG get_clip_config: resolved path={cfg.pretrained_model_path}")
    
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