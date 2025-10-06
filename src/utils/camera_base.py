"""
Base camera interface and common enums.

This module defines the abstract base class for all camera implementations
and common enums for camera models, platforms, and devices.
"""

from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image
from typing import Tuple


class CameraModel(Enum):
    """Supported camera models."""
    IMX500 = "imx500"
    IMX477 = "imx477"
    IMX219 = "imx219"


class Platform(Enum):
    """Supported platforms."""
    RPI = "rpi"
    JETSON = "jetson"


class Device(Enum):
    """Camera device identifiers."""
    CAMERA0 = "camera0"
    CAMERA1 = "camera1"


class BaseCamera(ABC):
    """
    Abstract base class for camera interfaces.

    Defines the common interface and attributes for all camera implementations,
    including model type, platform, and device identification.
    """

    def __init__(self, model: CameraModel, platform: Platform, device: Device):
        """
        Initialize base camera with hardware configuration.

        Args:
            model: Camera model type (IMX500, IMX477, IMX219)
            platform: Platform type (RPI, Jetson)
            device: Device identifier (camera0, camera1)
        """
        self.model = model
        self.platform = platform
        self.device = device

    @abstractmethod
    def capture_single_image(self) -> Tuple[str, Image.Image]:
        """
        Capture a single image from the camera.

        Returns:
            Tuple[str, Image.Image]: File path and PIL Image object
        """
        pass

    @property
    @abstractmethod
    def save_path(self) -> str:
        """Get the directory path for saving captured images."""
        pass