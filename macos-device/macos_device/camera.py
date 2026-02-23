"""Camera task using OpenCV."""

import cv2
import time
import logging
import numpy as np
from PIL import Image
from camera_framework import BaseTask
from typing import Optional

logger = logging.getLogger(__name__)


class Camera(BaseTask):
    """Capture frames from webcam or video file.
    
    Args:
        source: Camera index or video file path
        name: Task name
        fields: Field mappings (default output: "frame")
    
    Field mappings:
        frame: Output field for captured frame (default: "frame")
    """
    
    def __init__(self, source: int | str = 0, name: str = "Camera", fields: dict = None):
        super().__init__(name, fields)
        self.source = source
        self.cap: Optional[cv2.VideoCapture] = None
    
    def start(self):
        """Open camera/video source."""
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open camera source: {self.source}")
        
        # Wait for camera to initialize and try reading a few test frames
        time.sleep(1.5)
        
        # Warm up camera by reading and discarding a few frames
        for i in range(5):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.3)
        else:
            # If we still can't read after warm-up, raise error
            raise RuntimeError(f"Camera opened but cannot read frames from source: {self.source}")
    
    def process(self) -> None:
        """Capture a frame and write to output buffers."""
        if self.cap is None:
            self.start()
        
        ret, frame = self.cap.read()
        if not ret:
            logger.error(f"Camera read failed. cap={self.cap}, isOpened={self.cap.isOpened() if self.cap else False}")
            raise RuntimeError("Failed to read frame from camera")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Only create message if we have output buffers to write to
        if not self.outputs:
            return
        
        # Create message dict and add frame
        message = {}
        message.setdefault(self.field("frame"), []).append(frame_rgb)
        
        # Write to output buffers
        for buffer in self.outputs.values():
            buffer.put(message)
    
    def capture(self) -> Optional[Image.Image]:
        """Capture a single frame and return as PIL Image (for snapshot).
        
        Returns:
            PIL Image in RGB format, or None if capture fails
        """
        if self.cap is None or not self.cap.isOpened():
            self.start()
        
        # Try multiple times to read a frame
        for attempt in range(3):
            ret, frame = self.cap.read()
            if ret:
                break
            time.sleep(0.2)
        else:
            return None
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert numpy array to PIL Image
        return Image.fromarray(frame_rgb)
    
    def stop(self):
        """Release camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
