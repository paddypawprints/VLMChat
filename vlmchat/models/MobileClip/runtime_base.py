"""
Runtime base classes for CLIP image and text encoders.

This module defines abstract interfaces for CLIP backends, separating
image encoding and text encoding into independent components.
"""

from abc import abstractmethod
from typing import List
import torch
from PIL import Image

from models.model_base import BaseRuntime


class ClipImageRuntimeBase(BaseRuntime):
    """
    Abstract base class for CLIP image encoder runtime.
    
    Defines the interface for encoding images into feature vectors.
    Implementations handle model loading, preprocessing, and inference.
    """
    
    @property
    def max_batch_size(self) -> int:
        """
        Maximum batch size supported by this backend.
        
        Returns:
            Maximum number of images that can be encoded in a single call.
            Default is 1 (no batching). Subclasses should override if they support batching.
        """
        return 1
    
    @abstractmethod
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encodes a single PIL Image into a normalized feature tensor.
        
        Args:
            image: PIL Image to encode
            
        Returns:
            Normalized feature tensor of shape (1, feature_dim)
        """
        pass


class ClipTextRuntimeBase(BaseRuntime):
    """
    Abstract base class for CLIP text encoder runtime.
    
    Defines the interface for encoding text prompts into feature vectors.
    Implementations handle tokenization and text encoding.
    """
    
    @property
    def max_batch_size(self) -> int:
        """
        Maximum batch size supported by this backend.
        
        Returns:
            Maximum number of text prompts that can be encoded in a single call.
            Default is 1 (no batching). Subclasses should override if they support batching.
        """
        return 1
    
    @abstractmethod
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """
        Encodes a list of text strings into normalized feature tensors.
        
        Args:
            text_prompts: List of text strings to encode
            
        Returns:
            Normalized feature tensor of shape (len(text_prompts), feature_dim)
        """
        pass
