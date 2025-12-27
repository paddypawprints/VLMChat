"""
Ultralytics backend for YOLO inference.

This module provides a YoloRuntimeBase implementation using the Ultralytics library
for CPU-based inference. This is an alternative to the TensorRT backend for systems
without GPU support or when TensorRT is not available.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

try:
    from ultralytics import YOLO  # type: ignore[import-untyped]
    from ultralytics.engine.results import Boxes  # type: ignore[import-untyped]
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None  # type: ignore[assignment,misc,unused-ignore]
    Boxes = None  # type: ignore[assignment,misc,unused-ignore]

from vlmchat.models.Yolov8n.runtime_base import YoloRuntimeBase, Image
from vlmchat.utils.config import VLMChatConfig

logger = logging.getLogger(__name__)


class UltralyticsBackend(YoloRuntimeBase):
    """
    YOLO runtime implementation using Ultralytics library for CPU inference.
    
    This backend is designed for systems without GPU/TensorRT support or for
    development and testing purposes. It uses the official Ultralytics YOLO
    implementation which supports various model formats and devices.
    """
    
    def __init__(self, config: VLMChatConfig):
        """
        Initialize the Ultralytics backend.
        
        Args:
            config: Application configuration containing model settings
        """
        super().__init__(config)
        
        # Get model path from config or use default
        self.model_path = self._get_model_path()
        self.model: Optional[Any] = None  # YOLO instance when available
        self.class_names: List[str] = []
        self._is_ready = False
        
        if not ULTRALYTICS_AVAILABLE:
            logger.error("UltralyticsBackend Error: Ultralytics package not installed.")
            return
        
        if not self.model_path.exists():
            logger.error(f"YOLO model not found at: {self.model_path}")
            return
        
        try:
            logger.info(f"Loading YOLO model from: {self.model_path}")
            self.model = YOLO(str(self.model_path))  # type: ignore[misc]
            
            # Warm up the model with dummy inference
            dummy_image = np.uint8(np.zeros((640, 640, 3)))
            self.model(dummy_image, device='cpu', verbose=False)  # type: ignore[misc]
            
            # Extract class names
            if self.model and self.model.names:
                # model.names is a dict like {0: 'person', 1: 'bicycle', ...}
                self.class_names = [
                    self.model.names[i] for i in range(len(self.model.names))
                ]
            
            self._is_ready = True
            logger.info("UltralyticsBackend loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            self._is_ready = False
    
    @property
    def native_image_format(self) -> str:
        """YOLO Ultralytics backend expects NumPy arrays (BGR format)."""
        return "numpy"
    
    def _get_model_path(self) -> Path:
        """
        Get the model path from config or return default.
        
        Returns:
            Path to the YOLO model file
        """
        # Use the yolo_model_path from config
        if hasattr(self._config.model, 'yolo_model_path'):
            return self._config.model.yolo_model_path
        
        # Fallback to default
        return Path("~/yolov8n.pt").expanduser().absolute()
    
    @property
    def is_available(self) -> bool:
        """
        Check if the backend is ready for inference.
        
        Returns:
            True if model is loaded and ready, False otherwise
        """
        return self._is_ready and self.model is not None
    
    def prepare_image(self, image: Image) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Prepare image for Ultralytics inference.
        
        Note: Ultralytics handles preprocessing internally, so this method
        primarily passes through the image and returns metadata.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (image, scale_factor, metadata)
        """
        h, w = image.shape[:2]
        
        # Ultralytics handles resizing internally, so we just pass the image through
        # Scale is calculated based on the model's default input size (640)
        scale = max(h, w) / 640.0
        
        meta = {
            "original_height": h,
            "original_width": w,
            "mode": "ultralytics_internal"
        }
        
        return image, scale, meta
    
    def infer(self, blob: np.ndarray) -> np.ndarray:
        """
        Run inference using Ultralytics YOLO.
        
        Args:
            blob: Input image (Ultralytics handles preprocessing)
            
        Returns:
            Raw detections in a consistent format
        """
        if not self.is_available:
            raise RuntimeError("UltralyticsBackend is not ready.")
        
        try:
            # Run inference (Ultralytics returns Results objects)
            results = self.model(blob, device='cpu', verbose=False)  # type: ignore[union-attr]
            
            # Extract boxes from first result
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    
                    # Convert to numpy format: [x1, y1, x2, y2, conf, class_id]
                    detections = []
                    for box in boxes:
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().item()
                        cls = int(box.cls[0].cpu().item())
                        
                        # Stack as [x1, y1, x2, y2, conf, class_id]
                        detection = np.array([
                            xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf, cls
                        ])
                        detections.append(detection)
                    
                    if detections:
                        return np.stack(detections)
            
            # No detections found
            return np.zeros((0, 6), dtype=np.float32)
            
        except Exception as e:
            logger.error(f"Error during Ultralytics inference: {e}", exc_info=True)
            return np.zeros((0, 6), dtype=np.float32)
    
    def decode_output(
        self,
        raw_output: np.ndarray,
        scale: float,
        meta: Dict[str, Any],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Decode raw Ultralytics output into detection dictionaries.
        
        Args:
            raw_output: Raw detections from infer() [N, 6] where columns are
                       [x1, y1, x2, y2, confidence, class_id]
            scale: Scale factor from preprocessing
            meta: Metadata from preprocessing
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold (not used since Ultralytics handles NMS)
            
        Returns:
            List of detection dictionaries with bbox, confidence, class_id, class_name
        """
        if len(raw_output) == 0:
            return []
        
        detections = []
        for detection in raw_output:
            x1, y1, x2, y2, conf, cls_id = detection
            
            # Filter by confidence
            if conf < confidence_threshold:
                continue
            
            # Scale coordinates back to original image size
            bbox = [
                float(x1 * scale),
                float(y1 * scale),
                float(x2 * scale),
                float(y2 * scale)
            ]
            
            class_id = int(cls_id)
            class_name = (
                self.class_names[class_id] 
                if class_id < len(self.class_names) 
                else str(class_id)
            )
            
            detections.append({
                "bbox": bbox,
                "confidence": float(conf),
                "class_id": class_id,
                "class_name": class_name
            })
        
        return detections
