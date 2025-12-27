"""
Simplified YOLO Ultralytics backend for pipeline.

Self-contained implementation without heavy abstractions.
Returns Detection objects ready for pipeline use.
"""

import logging
import numpy as np
from typing import List
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

from ..detection import Detection
from ..cache.image import ImageContainer
from ..image.formats import ImageFormat
from ..categories import CocoCategory

logger = logging.getLogger(__name__)


class YoloUltralytics:
    """
    Simplified YOLO backend using Ultralytics library.
    
    No base classes, no runtime switching - just a simple interface
    for running YOLO detection and returning Detection objects.
    """
    
    def __init__(self, model_path: str = "~/yolov8n.pt"):
        """
        Initialize YOLO model.
        
        Args:
            model_path: Path to YOLO model file (.pt)
        """
        if not ULTRALYTICS_AVAILABLE:
            raise ImportError("Ultralytics package not installed. Install with: pip install ultralytics")
        
        self.model_path = Path(model_path).expanduser()
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"YOLO model not found at: {self.model_path}")
        
        # Load model
        logger.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))
        
        # Warm up with dummy inference
        dummy_image = np.uint8(np.zeros((640, 640, 3)))
        self.model(dummy_image, device='cpu', verbose=False)
        
        # Extract class names
        self.class_names = [
            self.model.names[i] for i in range(len(self.model.names))
        ] if self.model.names else []
        
        logger.info(f"YOLO model loaded: {len(self.class_names)} classes")
    
    @property
    def is_available(self) -> bool:
        """Check if model is ready."""
        return self.model is not None
    
    @property
    def preferred_format(self) -> ImageFormat:
        """YOLO expects numpy arrays in BGR format."""
        return ImageFormat.NUMPY
    
    def detect(
        self,
        image: ImageContainer,
        confidence: float = 0.25,
        iou: float = 0.45
    ) -> List[Detection]:
        """
        Run YOLO detection on an image.
        
        Args:
            image: ImageContainer to detect from
            confidence: Minimum confidence threshold (0.0 to 1.0)
            iou: IoU threshold for NMS (0.0 to 1.0)
            
        Returns:
            List of Detection objects (extends ImageContainer)
        """
        if not self.is_available:
            raise RuntimeError("YOLO model not loaded")
        
        # Get image as numpy array (BGR)
        numpy_image = image.get(ImageFormat.NUMPY)
        
        # Run inference (Ultralytics handles preprocessing and NMS)
        results = self.model(
            numpy_image,
            conf=confidence,
            iou=iou,
            device='cpu',
            verbose=False
        )
        
        # Convert to Detection objects
        detections = []
        
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                
                for box in boxes:
                    # Extract detection info
                    xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                    conf = float(box.conf[0].cpu().item())
                    cls = int(box.cls[0].cpu().item())
                    
                    # Skip if below threshold
                    if conf < confidence:
                        continue
                    
                    # Get category
                    category = CocoCategory.from_id(cls)
                    if category is None:
                        logger.warning(f"Unknown COCO class ID: {cls}, skipping detection")
                        continue
                    
                    # Create Detection object
                    detection = Detection(
                        bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        confidence=conf,
                        category=category,
                        source_image=image
                    )
                    
                    detections.append(detection)
        
        logger.debug(f"YOLO detected {len(detections)} objects")
        return detections
    
    def __str__(self) -> str:
        """String representation."""
        return f"YoloUltralytics(model={self.model_path.name}, classes={len(self.class_names)})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()
