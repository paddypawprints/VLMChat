"""
CLIP model wrapper with multiple runtime support.

This module provides:
- CLIPRuntimeBase: An abstract interface for CLIP backends.
- OpenClipBackend: A concrete backend using the 'open_clip' library.
- CLIPModel: The main facade that manages runtimes and exposes methods
             like `get_matches`.
"""

import logging
import torch
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from abc import abstractmethod

# --- Third-party imports ---
try:
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
# --- End third-party imports ---

from models.model_base import BaseModel, BaseRuntime
from utils.config import VLMChatConfig
from metrics.metrics_collector import Collector, null_collector
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Define CLIP-specific Runtime Interface ---

class CLIPRuntimeBase(BaseRuntime):
    """
    Abstract base class for a CLIP runtime.
    
    This defines the specific methods a CLIP backend must provide,
    in addition to the `is_available` property from BaseRuntime.
    """
    
    @abstractmethod
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encodes a single PIL Image into an image feature tensor.
        """
        pass
        
    @abstractmethod
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encodes a list of text strings into a text feature tensor.
        """
        pass

# --- Concrete Backend Implementations ---

class OpenClipBackend(CLIPRuntimeBase):
    """
    CLIP runtime implementation using the 'open_clip' library.
    """
    
    def __init__(self, config: VLMChatConfig):
        super().__init__(config)
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._is_ready = False
        
        if not OPEN_CLIP_AVAILABLE:
            logger.error("OpenClipBackend Error: 'open_clip' or 'mobileclip' library not installed.")
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
            
            logger.info(f"Loading OpenCLIP model: {model_name}")
            logger.info(f"Model path: {resolved_path}")
            
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name,
                pretrained=resolved_path,
                **model_kwargs
            )
            
            self.tokenizer = open_clip.get_tokenizer(model_name)
            
            # Model needs to be in eval mode
            self.model.eval()
            
            # Reparameterize if it's a MobileCLIP model
            if "MobileCLIP" in model_name:
                logger.info("Reparameterizing MobileCLIP model for inference...")
                self.model = reparameterize_model(self.model)
                
            self._is_ready = True
            logger.info("OpenClipBackend loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load OpenCLIP model: {e}", exc_info=True)
            self._is_ready = False
    
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
                            snapshot_dir = sorted(snapshot_dirs, key=lambda x: x.stat().st_mtime, reverse=True)[0]
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
                    model_path = hf_hub_download(
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

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encodes a single PIL Image."""
        if not self.is_available or self.preprocess is None or self.model is None:
            raise RuntimeError("OpenClipBackend is not ready.")
            
        preprocessed_image = self.preprocess(image.convert("RGB")).unsqueeze(0)
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            image_features = self.model.encode_image(preprocessed_image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
        return image_features

    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encodes a list of text strings."""
        if not self.is_available or self.tokenizer is None or self.model is None:
            raise RuntimeError("OpenClipBackend is not ready.")
            
        text = self.tokenizer(text_prompts)
        
        with torch.no_grad(), torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
            text_features = self.model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
        return text_features

# --- CLIP Model Facade ---

class CLIPModel(BaseModel):
    """
    Facade for the CLIP model.
    
    Manages runtime switching and provides a high-level `get_matches` API.
    """
    
    def __init__(self, config: VLMChatConfig, collector: Optional[Collector] = null_collector()):
        super().__init__(config, collector)
        
        # Cache for text prompt embeddings
        self._text_features_cache: Dict[str, torch.Tensor] = {}
        self._cached_prompt_list: List[str] = []
        
        # Load the default runtime
        self.set_runtime('auto') # set_runtime will call _make_runtime
        
    def _make_runtime(self, runtime_name: str) -> Tuple[BaseRuntime, str]:
        """
        Factory method for creating CLIP runtimes.
        """
        # 'auto' or 'open_clip' (default)
        if runtime_name == 'auto' or runtime_name == 'open_clip':
            backend = OpenClipBackend(self._config)
            if backend.is_available:
                return backend, 'open_clip'
            
            if runtime_name == 'open_clip':
                raise RuntimeError("OpenCLIP runtime requested but failed to load.")
        
        # TODO: Add logic for other backends
        # elif runtime_name == 'tensorrt':
        #     backend = TensorRTClipBackend(self._config)
        #     if backend.is_available:
        #         return backend, 'tensorrt'
        #     raise RuntimeError("TensorRT runtime requested but failed to load.")

        # Fallback error
        if runtime_name == 'auto':
            raise RuntimeError("No available CLIP runtime found.")
        else:
            raise RuntimeError(f"Unknown or unavailable CLIP runtime: {runtime_name}")

    def _runtime_as_clip(self) -> CLIPRuntimeBase:
        """Internal helper to safely cast and type-check the runtime."""
        if not isinstance(self._runtime, CLIPRuntimeBase):
            raise TypeError(f"Current runtime ({self._runtime.__class__.__name__}) does not implement CLIPRuntimeBase.")
        return self._runtime

    def pre_cache_text_prompts(self, prompts: List[str]) -> None:
        """
        Pre-encodes a list of text prompts and caches the features.
        This is useful for the semantic clusterer.
        """
        runtime = self._runtime_as_clip()
        logger.info(f"Pre-caching {len(prompts)} text prompts...")
        try:
            text_features = runtime.encode_text(prompts)
            
            # Clear old cache
            self._text_features_cache.clear()
            self._cached_prompt_list = prompts
            
            # Store one feature vector for each prompt
            for i, prompt in enumerate(prompts):
                self._text_features_cache[prompt] = text_features[i].unsqueeze(0) # Keep dims
                
            logger.info("Text prompts cached successfully.")
        except Exception as e:
            logger.error(f"Failed to pre-cache text prompts: {e}", exc_info=True)
            self._text_features_cache.clear()
            self._cached_prompt_list = []

    def get_matches(self, 
                    image: Image.Image, 
                    prompts: List[str]) -> List[Tuple[float, str]]:
        """
        Compares an image against a list of text prompts and returns
        a sorted list of (confidence, text) tuples.
        
        Returns:
            A list of (score, text) tuples, sorted by score.
        """
        runtime = self._runtime_as_clip()
        
        try:
            # 1. Encode the image
            image_features = runtime.encode_image(image)
            
            # 2. Encode the text
            # Check if we can use the cache
            if set(prompts) == set(self._cached_prompt_list):
                # Re-assemble the feature tensor from the cache in the correct order
                text_features = torch.cat([self._text_features_cache[p] for p in prompts], dim=0)
            else:
                # If prompts are different, encode them on-the-fly
                text_features = runtime.encode_text(prompts)
            
            # 3. Calculate similarity
            with torch.no_grad(), torch.cuda.amp.autocast():
                # We are returning raw cosine similarity scores, not probabilities.
                # Cosine similarity = dot product of normalized features
                # We multiply by 100 as per the original script's logic.
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
    #from utils.config import load_config

    #running from src directory
    
    # --- End Imports ---

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
                
                if clip_model.current_runtime() == 'none' or not isinstance(clip_model._runtime, OpenClipBackend):
                    raise RuntimeError("Failed to initialize OpenClipBackend.")
                
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