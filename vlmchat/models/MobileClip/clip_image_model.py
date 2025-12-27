"""
CLIP Image Model Facade.

This module provides a unified interface for CLIP image encoding
with multiple backend support (OpenCLIP, TensorRT).
"""

import logging
from typing import Optional, Tuple
from PIL import Image
import torch

from models.model_base import BaseModel, BaseRuntime
from utils.config import VLMChatConfig
from metrics.metrics_collector import Collector, null_collector

logger = logging.getLogger(__name__)


class ClipImageModel(BaseModel):
    """
    Facade for CLIP image encoding.
    
    Manages runtime selection and provides a simple `encode` API.
    """
    
    def __init__(self, config: VLMChatConfig, collector: Collector = null_collector()):
        super().__init__(config, collector)
        # Auto-select OpenCLIP runtime by default
        self.set_runtime("auto")
    
    def _make_runtime(self, runtime_name: str) -> Tuple[BaseRuntime, str]:
        """
        Create the appropriate CLIP image encoder runtime based on config.
        
        Args:
            runtime_name: Requested runtime ('openclip', 'tensorrt', or 'auto')
            
        Returns:
            Tuple of (runtime_instance, actual_runtime_name)
        """
        from .openclip_image_backend import OpenClipImageBackend
        from .tensorrt_image_backend import TensorRTImageBackend
        
        requested = runtime_name.lower().strip()
        
        if requested == "openclip":
            backend = OpenClipImageBackend(self._config)
            if backend.is_available:
                return backend, "openclip"
            else:
                raise RuntimeError("OpenCLIP image backend not available")
        
        elif requested == "tensorrt":
            backend = TensorRTImageBackend(self._config)
            if backend.is_available:
                return backend, "tensorrt"
            else:
                raise RuntimeError("TensorRT image backend not available")
        
        elif requested == "auto":
            # Try TensorRT first, fall back to OpenCLIP
            backend = TensorRTImageBackend(self._config)
            if backend.is_available:
                return backend, "tensorrt"
            
            backend = OpenClipImageBackend(self._config)
            if backend.is_available:
                return backend, "openclip"
            
            raise RuntimeError("No available CLIP image encoder found")
        
        else:
            raise RuntimeError(f"Unknown image backend: {runtime_name}")
    
    @property
    def max_batch_size(self) -> int:
        """
        Maximum batch size supported by the current backend.
        
        Returns:
            Maximum number of images that can be encoded in a single call.
            Returns 1 if no runtime is loaded.
        """
        if self._runtime and self._runtime.is_available:
            from .runtime_base import ClipImageRuntimeBase
            if isinstance(self._runtime, ClipImageRuntimeBase):
                return self._runtime.max_batch_size
        return 1
    
    def encode(self, image: Image.Image) -> torch.Tensor:
        """
        Encode a PIL Image into normalized image features.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Normalized image feature tensor
        """
        if not self._runtime or not self._runtime.is_available:
            raise RuntimeError("CLIP image encoder is not available")
        
        # Type narrowing: we know this is a ClipImageRuntimeBase
        from .runtime_base import ClipImageRuntimeBase
        if not isinstance(self._runtime, ClipImageRuntimeBase):
            raise TypeError(f"Runtime is not a ClipImageRuntimeBase: {type(self._runtime)}")
        
        return self._runtime.encode_image(image)
