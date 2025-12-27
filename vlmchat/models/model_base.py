"""
Defines the abstract base classes for models and their runtimes.

This module provides:
- BaseRuntime: An interface for a model's execution backend (e.g., ONNX).
- BaseModel: An interface for a model facade, managing runtime/backend-switching.
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple
from utils.config import VLMChatConfig
from metrics.metrics_collector import Collector, null_collector

logger = logging.getLogger(__name__)


class BaseRuntime(ABC):
    """
    Abstract base class for a model's runtime implementation.
    
    This class defines the minimal interface for a swappable backend.
    """
    def __init__(self, config: VLMChatConfig):
        self._config = config
        logger.info(f"Initializing runtime: {self.__class__.__name__}")

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """
        Returns True if this runtime is available and ready.
        e.g., for ONNX, this would check if the model file exists and ONNX Runtime is installed.
        """
        pass
    
    @property
    def native_image_format(self) -> Optional[str]:
        """
        Returns the image format this backend expects/prefers.
        
        Valid values: 'pil', 'numpy', 'torch_cpu', 'torch_gpu'
        Returns None if backend doesn't process images or has no preference.
        
        Tasks should provide images in this format for optimal performance.
        """
        return None  # Default: no image processing or no preference


class BaseModel(ABC):
    """
    Abstract base class for a model facade.
    
    This class manages the common logic for holding a model configuration,
    a metrics collector, and managing a swappable runtime backend.
    """
    
    def __init__(self, config: VLMChatConfig, collector: Collector = null_collector()):
        self._config = config
        self._collector = collector
        self._runtime: Optional[BaseRuntime] = None
        self._current_runtime_name: str = "none"

    @property
    def config(self) -> VLMChatConfig:
        """Get the model's configuration."""
        return self._config

    @property
    def collector(self) -> Collector:
        """Get the metrics collector."""
        return self._collector

    @abstractmethod
    def _make_runtime(self, runtime_name: str) -> Tuple[BaseRuntime, str]:
        """
        Factory method to create a runtime instance for this specific model type.
        
        This method must be implemented by child classes (e.g., SmolVLMModel, CLIPModel).
        It takes a requested runtime name (e.g., 'auto', 'onnx', 'tensorrt') and
        returns the instantiated runtime and the actual name of the runtime that
        was loaded (e.g., 'auto' might resolve to 'onnx').
        """
        pass

    def set_runtime(self, runtime: str) -> None:
        """
        Set the model runtime, recreating the backend.

        Args:
            runtime: The name of the runtime to use (e.g., 'onnx', 'tensorrt', 'hailo', 'auto').
        """
        requested_name = str(runtime).strip().lower()
        if not requested_name:
            raise ValueError("Runtime name cannot be empty.")

        try:
            new_runtime, actual_name = self._make_runtime(requested_name)
            
            if not new_runtime.is_available:
                raise RuntimeError(f"Runtime '{actual_name}' (requested '{requested_name}') is not available or failed to load.")
                
            self._runtime = new_runtime
            self._current_runtime_name = actual_name
            logger.info(f"{self.__class__.__name__} runtime set to: {self.current_runtime()}")
            
        except Exception as e:
            logger.error(f"Failed to set runtime to '{requested_name}': {e}")
            raise # Re-raise the exception

    def current_runtime(self) -> str:
        """Return the name of the currently selected runtime as a string."""
        return self._current_runtime_name