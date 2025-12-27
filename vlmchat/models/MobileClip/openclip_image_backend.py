"""
OpenCLIP image encoder backend.

This module provides the OpenCLIP implementation for CLIP image encoding.
"""

import logging
import torch
from typing import Optional
from PIL import Image
from pathlib import Path

# --- Third-party imports ---
try:
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model  # type: ignore[import-not-found]
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
# --- End third-party imports ---

from .runtime_base import ClipImageRuntimeBase
from utils.config import VLMChatConfig

logger = logging.getLogger(__name__)


class OpenClipImageBackend(ClipImageRuntimeBase):
    """
    CLIP image encoder using the 'open_clip' library.
    """
    
    def __init__(self, config: VLMChatConfig):
        super().__init__(config)
        self.model = None
        self.preprocess = None
        self._is_ready = False
        
        if not OPEN_CLIP_AVAILABLE:
            logger.error("OpenClipImageBackend Error: 'open_clip' or 'mobileclip' library not installed.")
            return

        try:
            # Get configuration from VLMChatConfig
            model_name = getattr(config.model, "clip_model_name", "MobileCLIP2-S0")
            pretrained_path = getattr(config.model, "clip_pretrained_path", None)
            model_kwargs = getattr(config.model, "clip_model_kwargs", None) or {}
            
            # Apply MobileCLIP-specific image normalization defaults
            if "MobileCLIP" in model_name:
                if 'image_mean' not in model_kwargs and 'image_std' not in model_kwargs:
                    if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
                        model_kwargs["image_mean"] = (0, 0, 0)
                        model_kwargs["image_std"] = (1, 1, 1)
            
            # Resolve model path: use config path if exists, otherwise download from HF
            resolved_path = self._resolve_model_path(model_name, pretrained_path)
            
            logger.info(f"Loading OpenCLIP image model: {model_name}")
            logger.info(f"Model path: {resolved_path}")
            
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(  # type: ignore[possibly-unbound]
                model_name,
                pretrained=resolved_path,
                **model_kwargs
            )
            
            # Model needs to be in eval mode
            self.model.eval()
            
            # Reparameterize if it's a MobileCLIP model
            if "MobileCLIP" in model_name:
                logger.info("Reparameterizing MobileCLIP image model for inference...")
                self.model = reparameterize_model(self.model)  # type: ignore[possibly-unbound]
                
            self._is_ready = True
            logger.info("OpenClipImageBackend loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load OpenCLIP image model: {e}", exc_info=True)
            self._is_ready = False
    
    @property
    def native_image_format(self) -> str:
        """OpenCLIP backend prefers PIL images."""
        return "pil"
    
    def _resolve_model_path(self, model_name: str, pretrained_path: Optional[str]) -> str:
        """
        Resolve the model path: use explicit path if provided and exists,
        otherwise check HuggingFace cache, then download from HuggingFace Hub.
        
        Args:
            model_name: Name of the CLIP model (e.g., "MobileCLIP2-S0")
            pretrained_path: Optional explicit path from config
            
        Returns:
            Resolved path to model file
        """
        # If explicit path provided and exists, use it
        if pretrained_path:
            p = Path(pretrained_path).expanduser()
            if p.exists():
                logger.info(f"Using explicit model path: {p}")
                return str(p)
            else:
                logger.info(f"Configured path does not exist: {pretrained_path}")
        
        # Check HuggingFace cache first (before attempting download)
        if "MobileCLIP" in model_name:
            hf_cache_base = Path.home() / ".cache" / "huggingface" / "hub"
            
            # Map model names to HF repo names
            repo_map = {
                "MobileCLIP2-S0": "models--apple--MobileCLIP2-S0",
                "MobileCLIP2-S1": "models--apple--MobileCLIP2-S1",
                "MobileCLIP2-S2": "models--apple--MobileCLIP2-S2",
                "MobileCLIP2-S3": "models--apple--MobileCLIP2-S3",
                "MobileCLIP2-S4": "models--apple--MobileCLIP2-S4",
                "MobileCLIP2-B": "models--apple--MobileCLIP2-B",
                "MobileCLIP2-L-14": "models--apple--MobileCLIP2-L-14",
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
                            # Use the most recent snapshot
                            sorted_dirs = sorted(snapshot_dirs, key=lambda x: x.stat().st_mtime, reverse=True)
                            snapshot_dir = sorted_dirs[0]
                            model_file = snapshot_dir / f"{model_name.lower().replace('-', '_')}.pt"
                            if model_file.exists():
                                logger.info(f"Found model in HuggingFace cache: {model_file}")
                                return str(model_file)
        
        # Download from HuggingFace Hub if available
        if HF_HUB_AVAILABLE and "MobileCLIP" in model_name:
            try:
                # Map model names to HF repos
                repo_map = {
                    "MobileCLIP2-S0": "apple/MobileCLIP2-S0",
                    "MobileCLIP2-S1": "apple/MobileCLIP2-S1",
                    "MobileCLIP2-S2": "apple/MobileCLIP2-S2",
                    "MobileCLIP2-S3": "apple/MobileCLIP2-S3",
                    "MobileCLIP2-S4": "apple/MobileCLIP2-S4",
                    "MobileCLIP2-B": "apple/MobileCLIP2-B",
                    "MobileCLIP2-L-14": "apple/MobileCLIP2-L-14",
                }
                
                repo_id = repo_map.get(model_name)
                if repo_id:
                    # Filename uses underscore, not hyphen
                    filename = f"{model_name.lower().replace('-', '_')}.pt"
                    logger.info(f"Downloading model from HuggingFace: {repo_id}/{filename}")
                    model_path = hf_hub_download(  # type: ignore[possibly-unbound]
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=None  # Use default HF cache
                    )
                    logger.info(f"Downloaded to: {model_path}")
                    return model_path
            except Exception as e:
                logger.warning(f"Could not download from HuggingFace: {e}")
        
        # Fall back to pretrained_path or default
        fallback = pretrained_path or f"./{model_name.lower()}.pt"
        logger.info(f"Using fallback path: {fallback}")
        return fallback

    @property
    def is_available(self) -> bool:
        """Returns True if the model loaded successfully."""
        return self._is_ready and self.model is not None
    
    @property
    def max_batch_size(self) -> int:
        """Returns the maximum batch size supported by OpenCLIP backend."""
        return 8

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encodes a single PIL Image into normalized image features."""
        if not self.is_available or self.preprocess is None or self.model is None:
            raise RuntimeError("OpenClipImageBackend is not ready.")
            
        preprocessed_image = self.preprocess(image.convert("RGB")).unsqueeze(0)  # type: ignore[operator,attr-defined]
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):  # type: ignore[attr-defined]
            image_features = self.model.encode_image(preprocessed_image)  # type: ignore[operator]
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features
