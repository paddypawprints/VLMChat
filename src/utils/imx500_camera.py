"""
IMX500 Camera implementation without object detection.

This module provides a basic IMX500 camera implementation that supports
image capture without neural processing capabilities.
"""

import os
from datetime import datetime
from PIL import Image
from typing import Tuple

from picamera2 import Picamera2
from picamera2.devices import IMX500

from .camera_base import BaseCamera, CameraModel, Platform, Device


class IMX500Camera(BaseCamera):
    """
    Basic IMX500 camera implementation without object detection.

    Provides standard camera functionality for the IMX500 sensor including
    image capture and basic configuration.
    """

    def __init__(self, platform: Platform = Platform.RPI, device: Device = Device.CAMERA0):
        """
        Initialize IMX500 camera without neural processing.

        Args:
            platform: Platform type (defaults to RPI)
            device: Device identifier (defaults to CAMERA0)
        """
        super().__init__(CameraModel.IMX500, platform, device)

        # Initialize IMX500 device
        self._imx500 = IMX500()

        # Initialize camera with RGB format
        self._picam2 = Picamera2(self._imx500.camera_num)
        main = {'format': 'RGB888'}
        self._config = self._picam2.create_preview_configuration(main)

        # Start camera
        self._picam2.start(self._config, show_preview=False)

        # Create directory for captured images using configuration
        from config import get_config
        config = get_config()
        self._save_path = config.paths.captured_images_dir
        if not os.path.exists(self._save_path):
            os.makedirs(self._save_path)

    @property
    def save_path(self) -> str:
        """Get the directory path for saving captured images."""
        return self._save_path

    def capture_single_image(self) -> Tuple[str, Image.Image]:
        """
        Capture a single image from the camera.

        Captures a single frame from the camera, converts it to a PIL Image,
        saves it to disk with a timestamp-based filename, and returns both
        the file path and image object.

        Returns:
            Tuple[str, Image.Image]: Tuple containing the saved file path and PIL Image object

        Raises:
            Exception: Camera capture or file I/O errors
        """
        # Capture raw array from camera
        array = self._picam2.capture_array()
        image = Image.fromarray(array)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"capture_{timestamp}.jpg"
        filepath = os.path.join(self._save_path, filename)

        # Save image to disk
        image.save(filepath)

        return filepath, image