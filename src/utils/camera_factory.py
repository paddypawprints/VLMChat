"""
Camera factory for creating camera instances.

This module provides a factory class for creating appropriate camera instances
based on model, platform, and device specifications.
"""

from typing import Union

from .camera_base import BaseCamera, CameraModel, Platform, Device
from config import get_config
import logging

logger = logging.getLogger(__name__)

# Per-platform default camera choices. Consumers can call
# CameraFactory.get_default_camera_for_platform(...) to inspect this mapping.
DEFAULT_CAMERA_BY_PLATFORM = {
    Platform.RPI: {
        "model": CameraModel.IMX500,
        "device": Device.CAMERA0,
        "with_detection": True,
    },
    Platform.JETSON: {
        "model": CameraModel.IMX219,
        "device": Device.CAMERA0,
        "with_detection": False,
    },
}
from .detection_base import ObjectDetectionInterface


class CameraFactory:
    """
    Factory class for creating camera instances.

    Provides static methods to create appropriate camera implementations
    based on hardware configuration and capability requirements.
    """

    @staticmethod
    def create_camera(
        model: CameraModel | None = None,
        platform: Platform | None = None,
        device: Device | None = None,
        with_detection: bool | None = None,
        args=None,
    ) -> BaseCamera:
        """
        Create a camera instance based on specifications.

        Args:
            model: Camera model type (IMX500, IMX477, IMX219)
            platform: Platform type (RPI, Jetson)
            device: Device identifier (camera0, camera1)
            with_detection: Whether to include object detection capabilities
            args: Optional arguments for detection configuration

        Returns:
            BaseCamera: Appropriate camera instance

        Raises:
            ValueError: If unsupported model/platform combination is requested
            NotImplementedError: If requested configuration is not yet implemented
        """
        # If platform isn't provided, use runtime detected platform from config
        if platform is None:
            cfg = get_config()
            detected = cfg.get_runtime_platform()
            platform = detected or Platform.RPI
        logger.debug(f"CameraFactory: runtime platform resolved to {platform}")

        # If model/device/with_detection are not provided, pick sensible defaults
        defaults = DEFAULT_CAMERA_BY_PLATFORM.get(platform, {})
        if model is None:
            model = defaults.get("model")
            logger.debug(f"CameraFactory: model not provided, using default {model} for platform {platform}")
        if device is None:
            device = defaults.get("device", Device.CAMERA0)
            logger.debug(f"CameraFactory: device not provided, using default {device} for platform {platform}")
        if with_detection is None:
            with_detection = defaults.get("with_detection", False)
            logger.debug(f"CameraFactory: with_detection not provided, using default {with_detection} for platform {platform}")

        # Now dispatch based on resolved values
        if platform == Platform.RPI and model == CameraModel.IMX500:
            if with_detection:
                logger.info(f"CameraFactory: creating IMX500 with detection on {platform} (device={device})")
                from .imx500_detection import IMX500ObjectDetection

                return IMX500ObjectDetection(args=args, platform=platform, device=device)
            else:
                logger.info(f"CameraFactory: creating IMX500 camera on {platform} (device={device})")
                from .imx500_camera import IMX500Camera

                return IMX500Camera(platform=platform, device=device)

        elif model == CameraModel.IMX477:
            # TODO: Implement IMX477 camera support
            raise NotImplementedError(f"IMX477 camera support not yet implemented")

        elif platform == Platform.JETSON and model == CameraModel.IMX219:
            logger.info(f"CameraFactory: creating IMX219 camera on {platform} (device={device})")
            from .imx219_camera import IMX219Camera
                
            return IMX219Camera(platform=platform, device=device)

        else:
            raise ValueError(f"Unsupported camera model: {model} on platform: {platform}")

    @staticmethod
    def create_detection_camera(
        model: CameraModel,
        platform: Platform | None = None,
        device: Device = Device.CAMERA0,
        args=None
    ) -> Union[BaseCamera, ObjectDetectionInterface]:
        """
        Create a camera instance with object detection capabilities.

        Convenience method that calls create_camera with with_detection=True.

        Args:
            model: Camera model type
            platform: Platform type
            device: Device identifier
            args: Optional arguments for detection configuration

        Returns:
            Camera instance with object detection capabilities

        Raises:
            ValueError: If model doesn't support object detection
        """
        if model != CameraModel.IMX500:
            raise ValueError(f"Object detection not supported for {model.value}")

        return CameraFactory.create_camera(
            model=model,
            platform=platform,
            device=device,
            with_detection=True,
            args=args
        )

    @staticmethod
    def get_supported_models() -> list[CameraModel]:
        """
        Get list of supported camera models.

        Returns:
            List of supported CameraModel enums
        """
        return [CameraModel.IMX500]  # TODO: Add IMX477, IMX219 when implemented

    @staticmethod
    def get_default_camera_for_platform(platform: Platform) -> dict:
        """Return the default camera mapping for the given platform."""
        return DEFAULT_CAMERA_BY_PLATFORM.get(platform, {}).copy()

    @staticmethod
    def supports_detection(model: CameraModel) -> bool:
        """
        Check if a camera model supports object detection.

        Args:
            model: Camera model to check

        Returns:
            True if model supports detection, False otherwise
        """
        return model == CameraModel.IMX500
