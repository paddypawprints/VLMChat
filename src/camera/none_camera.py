from .camera_base import BaseCamera, CameraModel
from ..utils.config import Device, VLMChatConfig
from ..metrics.metrics_collector import Collector
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
import os

class NoneCamera(BaseCamera):
    """A camera implementation that loads a test image from file."""

    def __init__(self, config: VLMChatConfig, collector: Collector):
        """Initialize the NoneCamera."""
        super().__init__(collector, CameraModel.NONE, config.platform, Device.CAMERA0)
        self.config = config
        self._collector = collector
        self.width = config.camera.width
        self.height = config.camera.height
        
        # Path to test image - relative to src/camera directory
        camera_dir = Path(__file__).parent
        self.test_image_path = camera_dir / "trail-riders.jpg"
    
    def capture_frame(self):
        """Return None as there is no camera to capture from."""
        return None
    
    def is_available(self):
        """Return True if test image exists."""
        return self.test_image_path.exists()
    
    def release(self):
        """Release resources (no-op for NoneCamera)."""
        pass

    def capture_single_image(self) -> Tuple[str, Image.Image]:
        """
        Load and resize trail_riders.jpg to match configured resolution.

        Returns:
            Tuple[str, Image.Image]: File path and PIL Image object
        """
        if not self.test_image_path.exists():
            raise FileNotFoundError(f"Test image not found: {self.test_image_path}")
        
        # Load image with opencv
        img_bgr = cv2.imread(str(self.test_image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {self.test_image_path}")
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize to match target resolution
        orig_height, orig_width = img_rgb.shape[:2]
        target_width, target_height = self.width, self.height
        
        # Calculate scaling factor
        scale_w = target_width / orig_width
        scale_h = target_height / orig_height
        
        # Use the larger scale to ensure we can crop (not pad on that dimension)
        scale = max(scale_w, scale_h)
        
        # Resize image
        new_width = int(orig_width * scale)
        new_height = int(orig_height * scale)
        
        if scale != 1.0:
            img_rgb = cv2.resize(img_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Center crop to exact dimensions
        current_height, current_width = img_rgb.shape[:2]
        
        # Only crop if image is LARGER than target
        if current_width > target_width:
            start_x = (current_width - target_width) // 2
            end_x = start_x + target_width
            img_rgb = img_rgb[:, start_x:end_x]
        
        if current_height > target_height:
            start_y = (current_height - target_height) // 2
            end_y = start_y + target_height
            img_rgb = img_rgb[start_y:end_y, :]
        
        # Pad with black if too small
        current_height, current_width = img_rgb.shape[:2]
        if current_width < target_width or current_height < target_height:
            pad_x = (target_width - current_width) // 2
            pad_y = (target_height - current_height) // 2
            
            img_rgb = cv2.copyMakeBorder(
                img_rgb,
                pad_y, target_height - current_height - pad_y,
                pad_x, target_width - current_width - pad_x,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0)
            )
        
        # Convert to PIL Image
        pil_image = Image.fromarray(img_rgb)
        
        return (str(self.test_image_path), pil_image)

    def save_path(self) -> str:
        return str(self.config.paths.captured_images_dir)