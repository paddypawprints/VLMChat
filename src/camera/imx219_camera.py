from .camera_base import BaseCamera, CameraModel, Platform, Device
import cv2
from PIL import Image
import numpy as np
import os
import datetime

def gstreamer_pipeline(
    capture_width=1920,
    capture_height=1080,
    display_width=1920,
    display_height=1080,
    framerate=30,
    flip_method=2,
):
    return (
    "nvarguscamerasrc ! "
    "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, format=(string)NV12, framerate=(fraction)%d/1 ! "
    "nvvidconv flip-method=%d ! "
    "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
    "videoconvert ! "
    "video/x-raw, format=(string)BGR ! appsink"
    % (
        capture_width,
        capture_height,
        framerate,
        flip_method,
        display_width,
        display_height,
    )
)

class IMX219Camera(BaseCamera):
    """Camera subclass for IMX219 on Jetson."""

    def __init__(
        self,
        model: CameraModel = CameraModel.IMX219,
        platform: Platform = Platform.JETSON,
        device: Device = Device.CAMERA0,
        capture_width=1920,
        capture_height=1080,
        display_width=1920,
        display_height=1080,
        framerate=30,
        flip_method=2,
        save_dir="./captures"
    ):
        super().__init__(model, platform, device)
        self.capture_width = capture_width
        self.capture_height = capture_height
        self.display_width = display_width
        self.display_height = display_height
        self.framerate = framerate
        self.flip_method = flip_method
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        self._cap = cv2.VideoCapture(
            gstreamer_pipeline(
                capture_width=self.capture_width,
                capture_height=self.capture_height,
                display_width=self.display_width,
                display_height=self.display_height,
                framerate=self.framerate,
                flip_method=self.flip_method,
            ),
            cv2.CAP_GSTREAMER,
        )
        if not self._cap.isOpened():
            raise RuntimeError("Unable to open IMX219 camera")

    def capture_single_image(self) -> tuple[str, Image.Image]:
        """
        Capture a single image from the camera.

        Returns:
            Tuple[str, Image.Image]: File path and PIL Image object
        """
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture image from IMX219 camera")

        # Convert to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        # Save image with timestamp and device info
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.device.value}_{timestamp}.jpg"
        filepath = os.path.join(self.save_path, filename)
        pil_image.save(filepath)

        return (filepath, pil_image)

    @property
    def save_path(self) -> str:
        """Get the directory path for saving captured images."""
        return self.save_dir

    def __del__(self):
        # Clean up camera resource on deletion
        if hasattr(self, '_cap') and self._cap.isOpened():
            self._cap.release()



