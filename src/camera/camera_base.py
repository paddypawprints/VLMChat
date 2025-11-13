"""
Base camera interface and common enums.

This module defines the abstract base class for all camera implementations
and common enums for camera models, platforms, and devices.
"""

from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image  # type: ignore
from typing import Tuple

from metrics.metrics_collector import Collector
from utils.platform_detect import Platform


class CameraModel(Enum):
    """Supported camera models."""
    IMX500 = "imx500"
    IMX477 = "imx477"
    IMX219 = "imx219"
    IMAGE_LIBRARY = "image_library"
    NONE = "none"


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

    def __init__(self, collector: Collector, model: CameraModel, platform: Platform, device: Device):
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
        self._collector = collector
        self._collector.register_timeseries("camera", ["inputs","generate"], ttl_seconds=600)

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