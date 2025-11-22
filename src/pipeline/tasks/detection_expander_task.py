"""
Detection Expander Task - Expands detection bounding boxes by a percentage.

This task takes detections and expands their bounding boxes to capture more
context around the detected objects. Useful for improving CLIP encoding by
including surrounding visual context.
"""

from typing import Dict
from PIL import Image

from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...object_detector.detection_base import Detection


@register_task('detection_expander')
class DetectionExpanderTask(BaseTask):
    """
    Expands detection bounding boxes by a configurable percentage.
    
    This task:
    1. Takes DETECTIONS from context
    2. Expands each detection's bounding box by the specified percentage
    3. Ensures expanded boxes stay within image boundaries
    4. Returns modified detections with expanded boxes
    
    The expansion helps capture contextual information around detected objects,
    which can improve semantic matching (e.g., seeing a person ON a horse
    instead of just a person in sitting posture).
    
    Example:
        expander = DetectionExpanderTask(
            expansion_factor=0.20,  # 20% expansion
            task_id="expander"
        )
        
        # After execution:
        # - Detection boxes are 20% larger in each direction
        # - Boxes are clipped to image boundaries
    """
    
    def __init__(self, expansion_factor: float = 0.20, task_id: str = "detection_expander"):
        """
        Initialize detection expander task.
        
        Args:
            expansion_factor: Factor to expand boxes (0.20 = 20% larger)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.expansion_factor = expansion_factor
        
        # Define contracts
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list
        }
        self.output_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.DETECTIONS: list
        }
    
    def configure(self, params: Dict[str, str]) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            params: Configuration parameters
                - expansion_factor: Expansion percentage (e.g., "0.20" for 20%)
        """
        if "expansion_factor" in params:
            self.expansion_factor = float(params["expansion_factor"])
    
    def _expand_detection_box(self, detection: Detection, image_width: int, image_height: int) -> Detection:
        """
        Expand a detection's bounding box.
        
        Args:
            detection: Detection to expand
            image_width: Image width for boundary checking
            image_height: Image height for boundary checking
            
        Returns:
            Detection with expanded box
        """
        x_min, y_min, x_max, y_max = detection.box
        
        # Calculate current dimensions
        width = x_max - x_min
        height = y_max - y_min
        
        # Calculate expansion amounts
        expand_x = width * self.expansion_factor
        expand_y = height * self.expansion_factor
        
        # Apply expansion
        new_x_min = max(0, x_min - expand_x)
        new_y_min = max(0, y_min - expand_y)
        new_x_max = min(image_width, x_max + expand_x)
        new_y_max = min(image_height, y_max + expand_y)
        
        # Create new detection with expanded box
        expanded_detection = Detection(
            box=(int(new_x_min), int(new_y_min), int(new_x_max), int(new_y_max)),
            object_category=detection.object_category,
            conf=detection.conf
        )
        
        # Preserve original ID
        expanded_detection.id = detection.id
        
        # Recursively expand children
        if detection.children:
            expanded_detection.children = [
                self._expand_detection_box(child, image_width, image_height)
                for child in detection.children
            ]
        
        return expanded_detection
    
    def run(self, context: Context) -> Context:
        """
        Expand detection bounding boxes.
        
        Args:
            context: Input context with IMAGE and DETECTIONS
            
        Returns:
            Context with expanded DETECTIONS
        """
        # Get image to determine boundaries
        image_list = context.data.get(ContextDataType.IMAGE)
        if not image_list:
            raise ValueError(f"Task {self.task_id}: IMAGE not found in context")
        
        image = image_list[0] if isinstance(image_list, list) else image_list
        
        # Convert to PIL Image if needed to get dimensions
        if isinstance(image, Image.Image):
            image_width, image_height = image.size
        else:
            # Assume numpy array with shape (height, width, channels)
            image_height, image_width = image.shape[:2]
        
        # Get detections from context
        detections = context.data.get(ContextDataType.DETECTIONS, [])
        
        if not detections:
            # No detections to expand
            return context
        
        # Expand each detection
        expanded_detections = []
        for det in detections:
            try:
                expanded_det = self._expand_detection_box(det, image_width, image_height)
                expanded_detections.append(expanded_det)
            except Exception as e:
                print(f"Warning: Failed to expand detection {det.id}: {e}")
                # Keep original detection on error
                expanded_detections.append(det)
        
        # Update context with expanded detections
        context.data[ContextDataType.DETECTIONS] = expanded_detections
        
        return context
    
    def __str__(self) -> str:
        """String representation."""
        return f"DetectionExpanderTask(id={self.task_id}, factor={self.expansion_factor:.2f})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()


if __name__ == "__main__":
    # Example usage
    print("\n--- DetectionExpanderTask Example ---\n")
    
    from PIL import Image
    import numpy as np
    
    # Create test detection
    test_det = Detection(
        id="test_1",
        object_category="person",
        conf=0.95,
        box=(100, 100, 200, 200)  # 100x100 box
    )
    
    # Create expander
    expander = DetectionExpanderTask(expansion_factor=0.20)
    
    # Create context
    ctx = Context()
    ctx.data[ContextDataType.IMAGE] = Image.new('RGB', (640, 480))
    ctx.data[ContextDataType.DETECTIONS] = [test_det]
    
    print(f"Original box: {test_det.box}")
    print(f"Expander: {expander}")
    
    # Run expander
    result_ctx = expander.run(ctx)
    expanded_det = result_ctx.data[ContextDataType.DETECTIONS][0]
    
    print(f"Expanded box: {expanded_det.box}")
    print(f"Original size: 100x100")
    new_width = expanded_det.box[2] - expanded_det.box[0]
    new_height = expanded_det.box[3] - expanded_det.box[1]
    print(f"Expanded size: {new_width}x{new_height}")
    print(f"Size increase: {((new_width/100 - 1) * 100):.1f}%")
