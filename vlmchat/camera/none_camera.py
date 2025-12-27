from .camera_base import BaseCamera, CameraModel
from utils.config import Device, VLMChatConfig
from metrics.metrics_collector import Collector
from PIL import Image
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import os
import requests
import tempfile

class NoneCamera(BaseCamera):
    """A camera implementation that loads a test image from file or URL."""

    def __init__(self, config: VLMChatConfig, collector: Collector, image_source: Optional[str] = None):
        """
        Initialize the NoneCamera.
        
        Args:
            config: VLMChat configuration
            collector: Metrics collector
            image_source: Optional file path or URL to image. If None, uses default trail-riders.jpg
        """
        super().__init__(collector, CameraModel.NONE, config.platform, Device.CAMERA0)
        self.config = config
        self._collector = collector
        self.width = config.camera.width
        self.height = config.camera.height
        
        # Determine image source
        if image_source:
            self.image_source = image_source
            self.is_url = image_source.startswith(('http://', 'https://'))
            self._downloaded_path = None  # Cache for downloaded images
        else:
            # Path to test image - relative to src/camera directory
            camera_dir = Path(__file__).parent
            self.image_source = str(camera_dir / "trail-riders.jpg")
            self.is_url = False
            self._downloaded_path = None
    
    def capture_frame(self):
        """Return None as there is no camera to capture from."""
        return None
    
    def is_available(self):
        """Return True if image source is accessible."""
        if self.is_url:
            return True  # Assume URL is available, will check on download
        else:
            return Path(self.image_source).exists()
    
    def release(self):
        """Release resources and clean up downloaded files."""
        if self._downloaded_path and os.path.exists(self._downloaded_path):
            try:
                os.remove(self._downloaded_path)
            except Exception:
                pass  # Ignore cleanup errors
    
    def _download_image(self, url: str) -> str:
        """
        Download image from URL to temporary file.
        
        Args:
            url: URL to download from
            
        Returns:
            Path to downloaded temporary file
            
        Raises:
            Exception: If download fails
        """
        if self._downloaded_path and os.path.exists(self._downloaded_path):
            return self._downloaded_path
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        # Save to temp file
        suffix = Path(url).suffix or '.jpg'
        fd, temp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        
        self._downloaded_path = temp_path
        return temp_path

    def capture_single_image(self) -> Tuple[str, Image.Image]:
        """
        Return current image from Environment if available, otherwise load from configured source.
        
        Checks for images stored by CameraTask or chat app in Environment.
        Supports both:
        - App+main+current_image (from chat app /load command)
        - CameraTask+*+current_image (from pipeline CameraTask instances)

        Returns:
            Tuple[str, Image.Image]: File path/URL and PIL Image object
        """
        # Try to get image from Environment first
        from vlmchat.pipeline.environment import Environment
        env = Environment.get_instance()
        
        # First check for chat app image (highest priority)
        current_image = env.get("App", "main", "current_image")
        if current_image is not None:
            return ("environment_image", current_image)
        
        # Then check for any CameraTask images
        for key in env.keys():
            if key.endswith("+current_image") and "CameraTask" in key:
                parts = key.split("+")
                if len(parts) == 3:
                    current_image = env.get(parts[0], parts[1], parts[2])
                    if current_image is not None:
                        return ("environment_image", current_image)
        
        # Fall back to configured image source
        if self.is_url:
            # Download from URL
            try:
                image_path = self._download_image(self.image_source)
            except Exception as e:
                raise ValueError(f"Failed to download image from {self.image_source}: {e}")
        else:
            # Use local file
            image_path = self.image_source
            if not Path(image_path).exists():
                raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image with opencv
        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
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
        
        return (self.image_source, pil_image)

    def save_path(self) -> str:
        return str(self.config.paths.captured_images_dir)