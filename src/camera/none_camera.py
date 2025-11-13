from .camera_base import BaseCamera, CameraModel
from utils.config import Device, VLMChatConfig
from metrics.metrics_collector import Collector
from PIL import Image
from typing import Tuple

class NoneCamera(BaseCamera):
    """A camera implementation that does nothing (null object pattern)."""

    def __init__(self, config: VLMChatConfig, collector: Collector):
        """Initialize the NoneCamera."""
        super().__init__(collector, CameraModel.NONE, config.platform, Device.CAMERA0)
        self.config = config
        self._collector = collector
    
    def capture_frame(self):
        """Return None as there is no camera to capture from."""
        return None
    
    def is_available(self):
        """Return False as this camera is not available."""
        return False
    
    def release(self):
        """Release resources (no-op for NoneCamera)."""
        pass

    def capture_single_image(self) -> Tuple[str, Image.Image]:
        """
        Capture a single image from the camera.

        Returns:
            Tuple[str, Image.Image]: File path and PIL Image object
        """
        pass

    def save_path(self) -> str:
        return self.config.captured_images_dir