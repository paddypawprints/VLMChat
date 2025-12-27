"""
YOLOv8 model wrapper with TensorRT runtime support.

This module provides:
- YoloModel: The main facade that manages runtimes and exposes detection methods.
"""

import logging
from typing import List, Dict, Any, Tuple

from vlmchat.models.model_base import BaseModel, BaseRuntime
from vlmchat.models.Yolov8n.runtime_base import YoloRuntimeBase, Image
from vlmchat.models.Yolov8n.tensorrt_backend import TensorRTBackend
from vlmchat.models.Yolov8n.ultralytics_backend import UltralyticsBackend
from vlmchat.utils.config import VLMChatConfig
from vlmchat.metrics.metrics_collector import Collector, null_collector

logger = logging.getLogger(__name__)


# --- YOLO Model Facade ---

class YoloModel(BaseModel):
    """
    Facade for the YOLOv8 model.
    
    Manages runtime switching and provides a high-level detection API.
    """
    
    def __init__(self, config: VLMChatConfig, collector: Collector = null_collector()):
        super().__init__(config, collector)
        # Auto-select TensorRT runtime
        self.set_runtime("tensorrt")
    
    def _make_runtime(self, runtime_name: str) -> Tuple[BaseRuntime, str]:
        """
        Factory method to create a YOLO runtime instance.
        
        Args:
            runtime_name: Requested runtime ('tensorrt' or 'auto')
            
        Returns:
            Tuple of (runtime_instance, actual_runtime_name)
        """
        requested = runtime_name.lower().strip()
        
        if requested == "tensorrt":
            backend = TensorRTBackend(self._config)
            if backend.is_available:
                return backend, "tensorrt"
            else:
                raise RuntimeError("TensorRT backend not available for YOLO")
        
        elif requested == "ultralytics":
            backend = UltralyticsBackend(self._config)
            if backend.is_available:
                return backend, "ultralytics"
            else:
                raise RuntimeError("Ultralytics backend not available for YOLO")
        
        elif requested == "auto":
            # Try TensorRT first, fall back to Ultralytics
            backend = TensorRTBackend(self._config)
            if backend.is_available:
                return backend, "tensorrt"
            
            backend = UltralyticsBackend(self._config)
            if backend.is_available:
                return backend, "ultralytics"
            
            raise RuntimeError("No YOLO backend available (tried TensorRT, Ultralytics)")
        
        raise ValueError(f"Unknown YOLO runtime: {runtime_name}")
    
    def detect(
        self,
        image: Image,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Performs object detection on an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detections with bbox, confidence, class_id, class_name
        """
        if self._runtime is None:
            raise RuntimeError("No runtime set. Call set_runtime() first.")
        
        if not isinstance(self._runtime, YoloRuntimeBase):
            raise TypeError(f"Current runtime ({self._runtime.__class__.__name__}) does not implement YoloRuntimeBase.")
        
        # Prepare image
        blob, scale, meta = self._runtime.prepare_image(image)
        
        # Run inference
        raw_output = self._runtime.infer(blob)
        
        # Decode detections
        detections = self._runtime.decode_output(
            raw_output,
            scale,
            meta,
            confidence_threshold,
            iou_threshold
        )
        
        return detections
