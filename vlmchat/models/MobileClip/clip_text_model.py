"""
CLIP Text Model Facade.

This module provides a unified interface for CLIP text encoding
with multiple backend support (OpenCLIP, TensorRT).
"""

import logging
from typing import List, Dict, Tuple
import torch

from models.model_base import BaseModel, BaseRuntime
from utils.config import VLMChatConfig
from metrics.metrics_collector import Collector, null_collector

logger = logging.getLogger(__name__)


class ClipTextModel(BaseModel):
    """
    Facade for CLIP text encoding.
    
    Manages runtime selection, provides a simple `encode` API,
    and supports text prompt caching.
    """
    
    def __init__(self, config: VLMChatConfig, collector: Collector = null_collector()):
        super().__init__(config, collector)
        self._cached_prompts: Dict[Tuple[str, ...], torch.Tensor] = {}
        # Auto-select OpenCLIP runtime by default
        self.set_runtime("auto")
    
    def _make_runtime(self, runtime_name: str) -> Tuple[BaseRuntime, str]:
        """
        Create the appropriate CLIP text encoder runtime based on config.
        
        Args:
            runtime_name: Requested runtime ('openclip', 'tensorrt', or 'auto')
            
        Returns:
            Tuple of (runtime_instance, actual_runtime_name)
        """
        from .openclip_text_backend import OpenClipTextBackend
        from .tensorrt_text_backend import TensorRTTextBackend
        
        requested = runtime_name.lower().strip()
        
        if requested == "openclip":
            backend = OpenClipTextBackend(self._config)
            if backend.is_available:
                return backend, "openclip"
            else:
                raise RuntimeError("OpenCLIP text backend not available")
        
        elif requested == "tensorrt":
            backend = TensorRTTextBackend(self._config)
            if backend.is_available:
                return backend, "tensorrt"
            else:
                raise RuntimeError("TensorRT text backend not available")
        
        elif requested == "auto":
            # Try TensorRT first, fall back to OpenCLIP
            backend = TensorRTTextBackend(self._config)
            if backend.is_available:
                return backend, "tensorrt"
            
            backend = OpenClipTextBackend(self._config)
            if backend.is_available:
                return backend, "openclip"
            
            raise RuntimeError("No available CLIP text encoder found")
        
        else:
            raise RuntimeError(f"Unknown text backend: {runtime_name}")
    
    @property
    def max_batch_size(self) -> int:
        """
        Maximum batch size supported by the current backend.
        
        Returns:
            Maximum number of text prompts that can be encoded in a single call.
            Returns 1 if no runtime is loaded.
        """
        if self._runtime and self._runtime.is_available:
            from .runtime_base import ClipTextRuntimeBase
            if isinstance(self._runtime, ClipTextRuntimeBase):
                return self._runtime.max_batch_size
        return 1
    
    def encode(self, text_prompts: List[str], use_cache: bool = False) -> torch.Tensor:
        """
        Encode a list of text prompts into normalized text features.
        
        Args:
            text_prompts: List of text strings to encode
            use_cache: If True, use cached features for already-encoded prompts
            
        Returns:
            Normalized text feature tensor
        """
        if not self._runtime or not self._runtime.is_available:
            raise RuntimeError("CLIP text encoder is not available")
        
        # Type narrowing: we know this is a ClipTextRuntimeBase
        from .runtime_base import ClipTextRuntimeBase
        if not isinstance(self._runtime, ClipTextRuntimeBase):
            raise TypeError(f"Runtime is not a ClipTextRuntimeBase: {type(self._runtime)}")
        
        if use_cache:
            # Check if all prompts are cached
            cache_key = tuple(text_prompts)
            if cache_key in self._cached_prompts:
                return self._cached_prompts[cache_key]
            
            # Encode and cache
            features = self._runtime.encode_text(text_prompts)
            self._cached_prompts[cache_key] = features
            return features
        else:
            return self._runtime.encode_text(text_prompts)
    
    def pre_cache_prompts(self, text_prompts: List[str]) -> None:
        """
        Pre-encode and cache text prompts for faster repeated queries.
        
        Args:
            text_prompts: List of text strings to pre-encode
        """
        if not self._runtime or not self._runtime.is_available:
            logger.warning("CLIP text encoder not available, cannot cache prompts")
            return
        
        # Type narrowing
        from .runtime_base import ClipTextRuntimeBase
        if not isinstance(self._runtime, ClipTextRuntimeBase):
            logger.warning(f"Runtime is not a ClipTextRuntimeBase: {type(self._runtime)}")
            return
        
        cache_key = tuple(text_prompts)
        if cache_key not in self._cached_prompts:
            logger.info(f"Caching {len(text_prompts)} text prompts")
            self._cached_prompts[cache_key] = self._runtime.encode_text(text_prompts)
    
    def clear_cache(self) -> None:
        """Clear all cached text prompt encodings."""
        self._cached_prompts.clear()
