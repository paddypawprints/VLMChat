"""
FashionClip model wrapper with multiple runtime support.

This module provides:
- FashionClipRuntimeBase: An abstract interface for FashionClip backends.
- FashionClipOpenClipBackend: A concrete backend using the 'open_clip' library with Marqo's FashionSigLIP.
- FashionClipModel: The main facade that manages runtimes and exposes methods like `get_matches`.
"""

import logging
import torch
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
from abc import abstractmethod

# --- Third-party imports ---
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
# --- End third-party imports ---

from models.model_base import BaseModel, BaseRuntime
from .fashion_clip_config import FashionClipConfig, get_fashion_clip_config
from utils.config import VLMChatConfig
from metrics.metrics_collector import Collector, null_collector

logger = logging.getLogger(__name__)

# --- Define FashionClip-specific Runtime Interface ---

class FashionClipRuntimeBase(BaseRuntime):
    """
    Abstract base class for a FashionClip runtime.
    
    This defines the specific methods a FashionClip backend must provide,
    in addition to the `is_available` property from BaseRuntime.
    """
    
    @abstractmethod
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encodes a single PIL Image into an image feature tensor.
        Features should be normalized.
        """
        pass
        
    @abstractmethod
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encodes a list of text strings into a text feature tensor.
        Features should be normalized.
        """
        pass

# --- Concrete Backend Implementations ---

class FashionClipOpenClipBackend(FashionClipRuntimeBase):
    """
    FashionClip runtime implementation using the 'open_clip' library with Marqo's FashionSigLIP model.
    """
    
    def __init__(self, config: VLMChatConfig):
        super().__init__(config)
        self._model_config: FashionClipConfig = get_fashion_clip_config(config)
        self.model = None
        self.preprocess_val = None
        self.tokenizer = None
        self._is_ready = False
        
        if not OPEN_CLIP_AVAILABLE:
            logger.error("FashionClipOpenClipBackend Error: 'open_clip' library not installed.")
            return

        try:
            logger.info(f"Loading FashionClip model: {self._model_config.model_name}")
            self.model, _, self.preprocess_val = open_clip.create_model_and_transforms(
                self._model_config.model_name,
                pretrained=self._model_config.pretrained_model_path if self._model_config.pretrained_model_path else None,
                **self._model_config.model_kwargs
            )
            
            self.tokenizer = open_clip.get_tokenizer(self._model_config.model_name)
            
            # Model needs to be in eval mode
            self.model.eval()
            
            self._is_ready = True
            logger.info("FashionClipOpenClipBackend loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load FashionClip model: {e}", exc_info=True)
            self._is_ready = False

    @property
    def is_available(self) -> bool:
        """Returns True if the model loaded successfully."""
        return self._is_ready and self.model is not None

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encodes a single PIL Image into normalized feature tensor.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Normalized image feature tensor with shape (1, feature_dim)
        """
        if not self.is_available or self.preprocess_val is None or self.model is None:
            raise RuntimeError("FashionClipOpenClipBackend is not ready.")
            
        preprocessed_image = self.preprocess_val(image.convert("RGB")).unsqueeze(0)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            image_features = self.model.encode_image(preprocessed_image, normalize=True)
            
        return image_features

    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encodes a list of text strings into normalized feature tensor.
        
        Args:
            text_prompts: List of text strings to encode
            
        Returns:
            Normalized text feature tensor with shape (len(text_prompts), feature_dim)
        """
        if not self.is_available or self.tokenizer is None or self.model is None:
            raise RuntimeError("FashionClipOpenClipBackend is not ready.")
            
        text = self.tokenizer(text_prompts)
        
        with torch.no_grad(), torch.cuda.amp.autocast():
            text_features = self.model.encode_text(text, normalize=True)
            
        return text_features

# --- FashionClip Model Facade ---

class FashionClipModel(BaseModel):
    """
    Facade for the FashionClip model.
    
    Manages runtime switching and provides a high-level `get_matches` API
    for comparing images against fashion-related text prompts.
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
        Factory method for creating FashionClip runtimes.
        
        Args:
            runtime_name: Name of the runtime to create ('auto', 'open_clip')
            
        Returns:
            Tuple of (runtime_instance, actual_runtime_name)
            
        Raises:
            RuntimeError: If the requested runtime cannot be created
        """
        # 'auto' or 'open_clip' (default and only option for now)
        if runtime_name == 'auto' or runtime_name == 'open_clip':
            backend = FashionClipOpenClipBackend(self._config)
            if backend.is_available:
                return backend, 'open_clip'
            
            if runtime_name == 'open_clip':
                raise RuntimeError("OpenCLIP runtime requested but failed to load.")
        
        # TODO: Add logic for other backends (TensorRT, ONNX, etc.)
        # elif runtime_name == 'tensorrt':
        #     backend = FashionClipTensorRTBackend(self._config)
        #     if backend.is_available:
        #         return backend, 'tensorrt'
        #     raise RuntimeError("TensorRT runtime requested but failed to load.")

        # Fallback error
        if runtime_name == 'auto':
            raise RuntimeError("No available FashionClip runtime found.")
        else:
            raise RuntimeError(f"Unknown or unavailable FashionClip runtime: {runtime_name}")

    def _runtime_as_fashion_clip(self) -> FashionClipRuntimeBase:
        """Internal helper to safely cast and type-check the runtime."""
        if not isinstance(self._runtime, FashionClipRuntimeBase):
            raise TypeError(f"Current runtime ({self._runtime.__class__.__name__}) does not implement FashionClipRuntimeBase.")
        return self._runtime

    def pre_cache_text_prompts(self, prompts: List[str]) -> None:
        """
        Pre-encodes a list of text prompts and caches the features.
        This is useful for repeated queries with the same prompt set.
        
        Args:
            prompts: List of text prompts to pre-encode and cache
        """
        runtime = self._runtime_as_fashion_clip()
        logger.info(f"Pre-caching {len(prompts)} fashion text prompts...")
        try:
            text_features = runtime.encode_text(prompts)
            
            # Clear old cache
            self._text_features_cache.clear()
            self._cached_prompt_list = prompts
            
            # Store one feature vector for each prompt
            for i, prompt in enumerate(prompts):
                self._text_features_cache[prompt] = text_features[i].unsqueeze(0) # Keep dims
                
            logger.info("Fashion text prompts cached successfully.")
        except Exception as e:
            logger.error(f"Failed to pre-cache text prompts: {e}", exc_info=True)
            self._text_features_cache.clear()
            self._cached_prompt_list = []

    def get_matches(self, 
                    image: Image.Image, 
                    prompts: List[str],
                    temperature: float = 100.0) -> List[Tuple[float, str]]:
        """
        Compares an image against a list of text prompts and returns
        a sorted list of (confidence, text) tuples.
        
        Uses cosine similarity between image and text embeddings, scaled
        by temperature and converted to probabilities via softmax.
        
        Args:
            image: PIL Image to compare against text prompts
            prompts: List of text prompts (e.g., ["a hat", "a t-shirt", "shoes"])
            temperature: Scaling factor for similarities (default: 100.0 to match original FashionClip)
            
        Returns:
            A list of (probability, text) tuples, sorted by probability (highest first)
            
        Example:
            >>> model = FashionClipModel(config)
            >>> image = Image.open("hat.jpg")
            >>> matches = model.get_matches(image, ["a hat", "a t-shirt", "shoes"])
            >>> print(matches)
            [(0.95, "a hat"), (0.03, "shoes"), (0.02, "a t-shirt")]
        """
        runtime = self._runtime_as_fashion_clip()
        
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
            
            # 3. Compute similarity (cosine similarity via dot product, since features are normalized)
            # image_features: (1, feature_dim)
            # text_features: (num_prompts, feature_dim)
            # Result: (1, num_prompts)
            similarities = (temperature * image_features @ text_features.T)
            
            # 4. Apply softmax to get probabilities
            probabilities = similarities.softmax(dim=-1)
            
            # 5. Convert to list of (score, text) tuples
            probs_list = probabilities[0].tolist()
            results = list(zip(probs_list, prompts))
            
            # 6. Sort by score (highest first)
            results.sort(reverse=True, key=lambda x: x[0])
            
            return results
            
        except Exception as e:
            logger.error(f"Failed to get matches: {e}", exc_info=True)
            raise

    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Directly encode an image to feature vector.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Normalized image feature tensor
        """
        runtime = self._runtime_as_fashion_clip()
        return runtime.encode_image(image)
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Directly encode text prompts to feature vectors.
        
        Args:
            text_prompts: List of text strings to encode
            
        Returns:
            Normalized text feature tensor
        """
        runtime = self._runtime_as_fashion_clip()
        return runtime.encode_text(text_prompts)
