"""
Color channel swap task for testing CLIP's color understanding.

This task swaps color channels (e.g., red<->blue) to test whether CLIP
follows actual pixel colors or relies on contextual understanding.

This task operates BEFORE ClipVisionTask by creating color-swapped detection crops
and storing them in the Detection objects for later use.
"""

import numpy as np
from PIL import Image
import logging
from typing import List, Tuple

from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...object_detector.detection_base import Detection

logger = logging.getLogger(__name__)


@register_task('color_swap')
class ColorSwapTask(BaseTask):
    """
    Swaps color channels in detection crops to test color attribute detection.
    
    This task:
    1. Takes IMAGE and DETECTIONS from context
    2. Crops each detection from the image
    3. Swaps specified color channels (e.g., swap_channels=(0, 2) swaps R and B)
    4. Stores swapped crops in detection.metadata['enhanced_crop']
    5. Sets detection.metadata['use_enhanced_crop'] = True
    
    The ClipVisionTask will check for 'enhanced_crop' and use it if present.
    
    Example:
        # Swap red and blue channels
        swap_task = ColorSwapTask(
            swap_channels=(0, 2),  # (R, B)
            task_id="rb_swap"
        )
    """
    
    def __init__(self, task_id: str = "color_swap", 
                 swap_channels: Tuple[int, int] = (0, 2)):
        """
        Initialize color swap task.
        
        Args:
            task_id: Unique identifier for this task
            swap_channels: Tuple of channel indices to swap (0=R, 1=G, 2=B)
                          Default (0, 2) swaps red and blue
        """
        super().__init__(task_id)
        self.swap_channels = swap_channels
        
        # Define contracts
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list
        }
        self.output_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list
        }
        
        channel_names = {0: "R", 1: "G", 2: "B"}
        ch1_name = channel_names.get(swap_channels[0], "?")
        ch2_name = channel_names.get(swap_channels[1], "?")
        
        logger.info(f"ColorSwapTask '{task_id}' initialized: "
                   f"swap {ch1_name}<->{ch2_name}")
    
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
        Swap color channels in detection crops.
        
        Args:
            context: Pipeline context with IMAGE and DETECTIONS
        
        Returns:
            Context with color-swapped crops stored in detection metadata
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
        
        swapped_count = 0
        ch1, ch2 = self.swap_channels
        
        for detection in detections:
            try:
                # Crop detection from image
                crop = self._crop_detection_from_image(image, detection)
                
                # Convert to numpy array
                crop_array = np.array(crop)
                
                # Swap channels
                if ch1 < crop_array.shape[2] and ch2 < crop_array.shape[2]:
                    swapped_array = crop_array.copy()
                    swapped_array[:, :, ch1] = crop_array[:, :, ch2]
                    swapped_array[:, :, ch2] = crop_array[:, :, ch1]
                    
                    # Convert back to PIL Image
                    swapped_crop = Image.fromarray(swapped_array)
                    
                    # Store in detection metadata
                    if not hasattr(detection, 'metadata') or detection.metadata is None:
                        detection.metadata = {}
                    
                    detection.metadata['enhanced_crop'] = swapped_crop
                    detection.metadata['use_enhanced_crop'] = True
                    detection.metadata['swapped_channels'] = self.swap_channels
                    
                    swapped_count += 1
                    
                    logger.debug(f"Task '{self.task_id}': Swapped channels {ch1}<->{ch2} "
                               f"for detection {detection.object_category}")
                else:
                    logger.warning(f"Task '{self.task_id}': Invalid channel indices, skipping detection")
                    
            except Exception as e:
                logger.error(f"Task '{self.task_id}': Failed to swap channels for detection: {e}")
                continue
        
        logger.info(f"Task '{self.task_id}': Swapped channels in {swapped_count}/{len(detections)} detections")
        
        return context
    
    def configure(self, **kwargs) -> None:
        """
        Configure task from parameters.
        
        Args:
            **kwargs: Configuration parameters with optional keys:
                - swap_channels: tuple of two ints (0-2)
        """
        if 'swap_channels' in kwargs:
            self.swap_channels = tuple(kwargs['swap_channels'])
            logger.info(f"Task '{self.task_id}': Updated swap_channels={self.swap_channels}")

