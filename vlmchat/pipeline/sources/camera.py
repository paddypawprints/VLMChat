"""
Base camera source for capturing frames from video devices.

This is an abstract base class. Use platform-specific implementations:
- JetsonCameraSource: For Jetson with GStreamer and buffer pool
- Add other implementations as needed
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logging.warning("OpenCV (cv2) not available - CameraSource will not work")

from .base import StreamSource

logger = logging.getLogger(__name__)


class CameraSource(StreamSource, ABC):
    """
    Abstract base class for camera sources.
    
    Defines the interface for camera sources. Subclasses implement
    platform-specific capture logic (GStreamer, V4L2, DirectShow, etc.)
    
    For Jetson with zero-copy buffer pool, use JetsonCameraSource.
    """
    
    def __init__(self, name: str, device: int = 0, fps: float = 30.0,
                 buffer_size: int = 300, width: Optional[int] = None,
                 height: Optional[int] = None, output_label: str = "frame"):
        """
        Initialize camera source.
        
        Args:
            name: Unique identifier for this source
            device: Camera device index (0 for default camera)
            fps: Target frames per second
            buffer_size: Number of frames to keep in ring buffer
            width: Optional frame width (None = camera default)
            height: Optional frame height (None = camera default)
            output_label: Label to use when adding frames to context (default: "frame")
        """
        super().__init__(name, buffer_size, output_label)
        
        if not CV2_AVAILABLE:
            raise RuntimeError("OpenCV (cv2) is required for CameraSource")
        
        self.device = device
        self.fps = fps
        self.target_width = width or 1920
        self.target_height = height or 1080
        
        # Frame timing
        self.frame_interval = 1.0 / fps if fps > 0 else 0
        
        # OpenCV capture object (set by subclass)
        self.capture: Optional[cv2.VideoCapture] = None
        
        logger.info(f"CameraSource '{name}' base initialized: device={device}, fps={fps}, buffer={buffer_size}")
    
    @abstractmethod
    def start(self) -> None:
        """
        Start camera capture. Subclasses must implement.
        
        Should open camera device and start continuous capture.
        """
        pass
    
    @abstractmethod  
    def stop(self) -> None:
        """
        Stop camera capture. Subclasses must implement.
        
        Should stop capture and release camera device.
        """
        pass
    
    @abstractmethod
    def poll(self) -> bool:
        """
        Check if new frame available. Subclasses must implement.
        
        Returns:
            True if new data available in ring buffer
        """
        pass
