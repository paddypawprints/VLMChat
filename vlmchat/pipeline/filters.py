"""
Image filters for detection regions.

Filters can be applied to detection bounding boxes for privacy
(blur faces), visualization effects, or other image transformations.
"""
from abc import ABC, abstractmethod
from typing import Tuple
import numpy as np


class ImageFilter(ABC):
    """
    Base class for image filters applied to detection regions.
    
    Filters operate on a rectangular region (bbox) within an image,
    modifying the pixels in-place. Used for privacy features like
    face blurring or visualization effects.
    """
    
    @abstractmethod
    def apply(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Apply filter to bbox region of image.
        
        Args:
            image: Full image (numpy array, HxWxC, RGB)
            bbox: (x1, y1, x2, y2) region to filter (in image coordinates)
        
        Returns:
            Image with filter applied to bbox region (modifies in-place and returns)
        """
        pass


class NoOpFilter(ImageFilter):
    """
    No-operation filter - does nothing.
    
    Useful as placeholder or for testing.
    """
    
    def apply(self, image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Apply no-op (returns image unchanged)."""
        return image
