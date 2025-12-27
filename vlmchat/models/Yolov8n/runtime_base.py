"""
Runtime base interface for YOLO backends.

Defines the abstract methods that YOLO backends must implement.
Extends BaseRuntime from model_base with YOLO-specific methods.
"""
from abc import abstractmethod
from typing import List, Dict, Any, Tuple
import numpy as np

from vlmchat.models.model_base import BaseRuntime

# Type alias for clarity
Image = np.ndarray


class YoloRuntimeBase(BaseRuntime):
    """
    Abstract base class for a YOLO runtime.
    
    This defines the specific methods a YOLO backend must provide,
    in addition to the `is_available` property from BaseRuntime.
    """
    
    @abstractmethod
    def prepare_image(self, image: Image) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Prepares an image for inference by resizing and normalizing.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (preprocessed_blob, scale, metadata)
        """
        pass
    
    @abstractmethod
    def infer(self, blob: np.ndarray) -> np.ndarray:
        """
        Runs inference on the preprocessed image blob.
        
        Args:
            blob: Preprocessed image tensor
            
        Returns:
            Raw model output
        """
        pass
    
    @abstractmethod
    def decode_output(
        self,
        raw_output: np.ndarray,
        scale: float,
        meta: Dict[str, Any],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Decodes raw model output into detection boxes.
        
        Args:
            raw_output: Raw output from model inference
            scale: Scale factor from preprocessing
            meta: Metadata from preprocessing
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detection dictionaries with keys: bbox, confidence, class_id, class_name
        """
        pass
