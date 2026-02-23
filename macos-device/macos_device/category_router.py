"""YOLO category filter - filters detections by category mask."""

from typing import Optional
from camera_framework import BaseTask
from camera_framework.detection import Detection, CocoCategory, ImageFormat


class YoloCategoryRouter(BaseTask):
    """Filters YOLO detections using shared DetectionFilter.
    
    Uses boolean vector from DetectionFilter to filter by COCO category.
    Thread-safe for MQTT updates via shared filter_config.
    
    Example:
        from .detection_filter import DetectionFilter
        
        filter_config = DetectionFilter()
        # Enable only persons (category.id == 0)
        filter_config.set_category_mask([i == 0 for i in range(80)])
        
        router = YoloCategoryRouter(filter_config=filter_config)
        router.add_input("detections", yolo_buffer)
        router.add_output("filtered", attributes_buffer)
    """
    
    def __init__(
        self, 
        name: str = "yolo_category_router",
        filter_config: Optional['DetectionFilter'] = None
    ):
        super().__init__(name=name)
        
        # Shared filter configuration (thread-safe)
        self.filter_config = filter_config
    
    def process(self) -> None:
        """Filter detections by category mask."""
        if not self.inputs or not self.filter_config:
            return
        
        input_buffer = list(self.inputs.values())[0]
        message = input_buffer.get()
        if not message:
            return
        
        detections = message.get("detections", [])
        
        # Filter by category mask
        filtered = [
            d for d in detections 
            if self.filter_config.matches_category(d.category)
        ]
        
        # If all detections filtered out, create synthetic full-frame detection
        # This ensures downstream buffers (especially snapshot) always have an image
        if not filtered and detections:
            # Get source image from first detection
            source_detection = detections[0]
            if source_detection.source_image:
                width = source_detection.source_image.width if hasattr(source_detection.source_image, 'width') else source_detection.source_image.size[0]
                height = source_detection.source_image.height if hasattr(source_detection.source_image, 'height') else source_detection.source_image.size[1]
                synthetic = Detection(
                    bbox=(0.0, 0.0, float(width), float(height)),
                    confidence=0.0,  # Low confidence ensures no alerts
                    category=CocoCategory.UNKNOWN,
                    source_image=source_detection.source_image,
                    source_format=source_detection.source_format or ImageFormat.PIL
                )
                filtered.append(synthetic)
        
        # Write to output if any detections (including synthetic)
        if filtered and self.outputs:
            output_buffer = list(self.outputs.values())[0]
            message = {"detections": filtered}
            output_buffer.put(message)
