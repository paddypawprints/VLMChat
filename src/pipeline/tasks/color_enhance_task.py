"""
Color enhancement task for improving attribute detection.

This task enhances color saturation in detection crops to help CLIP
better distinguish color-based attributes like "red shirt", "blue hat", etc.

This task operates BEFORE ClipVisionTask by creating enhanced detection crops
and storing them in the Detection objects for later use.
"""

import numpy as np
from PIL import Image, ImageEnhance
import logging
from typing import List

from ..task_base import BaseTask, Context, ContextDataType
from ...object_detector.detection_base import Detection

logger = logging.getLogger(__name__)


class ColorEnhanceTask(BaseTask):
    """
    Enhances color saturation in detection crops for improved color attribute detection.
    
    This task:
    1. Takes IMAGE and DETECTIONS from context
    2. Crops each detection from the image
    3. Enhances color saturation by a configurable factor
    4. Stores enhanced crops in detection.metadata['enhanced_crop']
    5. Sets detection.metadata['use_enhanced_crop'] = True
    
    The ClipVisionTask will check for 'enhanced_crop' and use it if present.
    
    Example:
        enhance_task = ColorEnhanceTask(
            saturation_factor=2.5,  # 2.5x saturation boost
            task_id="color_enhance"
        )
    """
    
    def __init__(self, task_id: str = "color_enhance", 
                 saturation_factor: float = 2.5):
        """
        Initialize color enhancement task.
        
        Args:
            task_id: Unique identifier for this task
            saturation_factor: Saturation multiplier (1.0 = no change, 2.5 = strong boost)
        """
        super().__init__(task_id)
        self.saturation_factor = saturation_factor
        
        # Define contracts
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list
        }
        self.output_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list
        }
        
        logger.info(f"ColorEnhanceTask '{task_id}' initialized: "
                   f"saturation_factor={saturation_factor}")
    
    def _crop_detection_from_image(self, image: Image.Image, detection: Detection) -> Image.Image:
        """
        Crop image region for a detection box.
        
        Args:
            image: PIL Image
            detection: Detection with bounding box
            
        Returns:
            Cropped image as PIL Image
        """
        x1, y1, x2, y2 = detection.box
        
        # Ensure coordinates are within image bounds
        x1 = max(0, int(x1))
        y1 = max(0, int(y1))
        x2 = min(image.width, int(x2))
        y2 = min(image.height, int(y2))
        
        # Crop
        cropped = image.crop((x1, y1, x2, y2))
        return cropped
    
    def run(self, context: Context) -> Context:
        """
        Enhance color saturation in detection crops.
        
        Args:
            context: Pipeline context with IMAGE and DETECTIONS
        
        Returns:
            Context with enhanced crops stored in detection metadata
        """
        # Get image from context
        image_list = context.data.get(ContextDataType.IMAGE)
        if not image_list:
            logger.warning(f"Task '{self.task_id}': No image in context")
            return context
        
        image = image_list[0] if isinstance(image_list, list) else image_list
        
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Get detections
        detections = context.data.get(ContextDataType.DETECTIONS, [])
        if not detections:
            logger.warning(f"Task '{self.task_id}': No detections in context")
            return context
        
        enhanced_count = 0
        for detection in detections:
            try:
                # Crop detection from image
                crop = self._crop_detection_from_image(image, detection)
                
                # Enhance saturation
                enhancer = ImageEnhance.Color(crop)
                enhanced_crop = enhancer.enhance(self.saturation_factor)
                
                # Store in detection metadata
                if not hasattr(detection, 'metadata') or detection.metadata is None:
                    detection.metadata = {}
                
                detection.metadata['enhanced_crop'] = enhanced_crop
                detection.metadata['use_enhanced_crop'] = True
                detection.metadata['saturation_factor'] = self.saturation_factor
                
                enhanced_count += 1
                
                logger.debug(f"Task '{self.task_id}': Enhanced detection "
                           f"{detection.object_category} with {self.saturation_factor}x saturation")
                
            except Exception as e:
                logger.error(f"Task '{self.task_id}': Failed to enhance detection: {e}")
                continue
        
        logger.info(f"Task '{self.task_id}': Enhanced {enhanced_count}/{len(detections)} detections")
        
        return context
    
    def configure(self, params: dict) -> None:
        """
        Configure task from parameters.
        
        Args:
            params: Configuration dict with optional keys:
                - saturation_factor: float (typically 1.0-3.0)
        """
        if 'saturation_factor' in params:
            self.saturation_factor = float(params['saturation_factor'])
            logger.info(f"Task '{self.task_id}': Updated saturation_factor={self.saturation_factor}")

