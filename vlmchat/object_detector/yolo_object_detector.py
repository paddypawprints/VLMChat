"""
Unified YOLO object detector using the models.Yolov8n module.

This detector wraps YoloModel and adapts it to the ObjectDetector interface,
supporting both TensorRT and Ultralytics backends.
"""

import logging
import numpy as np
from typing import List, Optional
from PIL import Image  # type: ignore[attr-defined]

from .detection_base import ObjectDetector, Detection
from .coco_categories import CocoCategory
from models.Yolov8n.yolo_model import YoloModel
from utils.config import VLMChatConfig

logger = logging.getLogger(__name__)


class YoloObjectDetector(ObjectDetector):
    """
    ObjectDetector implementation using YoloModel from models.Yolov8n.
    
    This detector can use either TensorRT or Ultralytics backend based on
    configuration or runtime availability. It wraps YoloModel and converts
    its output to Detection objects compatible with the object_detector pipeline.
    """
    
    def __init__(
        self,
        config: VLMChatConfig,
        runtime: str = "auto",
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        source: Optional[ObjectDetector] = None
    ):
        """
        Initialize the YOLO object detector.
        
        Args:
            config: Application configuration
            runtime: Backend to use ('tensorrt', 'ultralytics', or 'auto')
            confidence_threshold: Minimum confidence for detections (default: 0.25)
            iou_threshold: IoU threshold for NMS (default: 0.45)
            source: Optional preceding detector in the pipeline
        """
        super().__init__(source)
        
        self.config = config
        self.runtime_name = runtime
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        self.model: Optional[YoloModel] = None
        self._ready = False
    
    def start(self, audit: bool = False) -> None:
        """
        Load the YOLO model and prepare it for inference.
        
        Args:
            audit: Optional audit flag passed to source detector
        """
        super().start(audit)  # Start the source, if any
        
        if self._ready:
            return
        
        try:
            logger.info(f"Loading YOLO model with runtime: {self.runtime_name}")
            self.model = YoloModel(self.config)
            self.model.set_runtime(self.runtime_name)
            
            self._ready = True
            actual_runtime = self.model.current_runtime()
            logger.info(f"YOLO model loaded successfully using {actual_runtime} backend")
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}", exc_info=True)
            self.model = None
            self._ready = False
    
    def stop(self) -> None:
        """
        Clean up and unload the model.
        """
        super().stop()  # Stop the source, if any
        
        if self.model is not None:
            # YoloModel cleanup if needed
            self.model = None
        
        self._ready = False
        logger.info("YOLO detector stopped")
    
    def readiness(self) -> bool:
        """
        Check if this detector is ready AND its source (if any) is ready.
        
        Returns:
            True if detector is ready for inference
        """
        source_ready = self._source.readiness() if self._source else True
        return self._ready and self.model is not None and source_ready
    
    def get_labels(self) -> List[str]:
        """
        Get the list of class labels from this detector AND its source.
        
        Returns:
            Combined list of labels with duplicates removed
        """
        # Get labels from source
        source_labels = super().get_labels()
        
        # YOLO detects COCO 80 classes
        # We could load these from CocoCategory, but for now just note they exist
        yolo_labels = []  # Would populate from model if available
        
        # Combine, avoiding duplicates
        combined = source_labels + [l for l in yolo_labels if l not in source_labels]
        return combined
    
    def _detect_internal(  # type: ignore[override]
        self,
        image: Image,  # type: ignore[valid-type]
        detections: List[Detection]
    ) -> List[Detection]:
        """
        Perform object detection and add results to the detection list.
        
        Args:
            image: The input PIL Image
            detections: List of detections from the source (if any)
            
        Returns:
            Updated list of detections including YOLO detections
        """
        if not self._ready or self.model is None:
            # Model isn't ready, just return existing detections
            return detections
        
        try:
            # Convert PIL Image to numpy array (BGR format expected by YOLO)
            img_array = np.array(image)
            
            # Handle RGB vs BGR conversion if needed
            # PIL Images are RGB, but OpenCV/YOLO typically expects BGR
            if img_array.ndim == 3 and img_array.shape[2] == 3:
                # Convert RGB to BGR
                img_array = img_array[:, :, ::-1]
            
            # Run YOLO detection
            yolo_detections = self.model.detect(
                img_array,
                confidence_threshold=self.confidence_threshold,
                iou_threshold=self.iou_threshold
            )
            
            # Convert YOLO detections to Detection objects
            for yolo_det in yolo_detections:
                bbox = yolo_det['bbox']  # [x1, y1, x2, y2]
                
                # Convert bbox to integer tuple (x1, y1, x2, y2)
                box = (
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3])
                )
                
                # Get class name and resolve numeric IDs to COCO labels
                class_name = yolo_det['class_name']
                if class_name.isdigit():
                    # TensorRT backend may return numeric class IDs
                    category = CocoCategory.from_id(int(class_name))
                    class_name = category.label if category else class_name
                
                detection = Detection(
                    box=box,
                    object_category=class_name,
                    conf=yolo_det['confidence']
                )
                
                detections.append(detection)
            
        except Exception as e:
            logger.error(f"Error during YOLO detection: {e}", exc_info=True)
        
        return detections


# For backwards compatibility, create aliases
YoloDetector = YoloObjectDetector
YoloV8Detector = YoloObjectDetector
