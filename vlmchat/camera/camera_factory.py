"""
Camera factory for creating camera instances.
"""

import logging
from typing import Optional, Any

from metrics.metrics_collector import Collector

# Assuming 'utils' and 'camera' are sibling directories
from .none_camera import NoneCamera
from utils.platform_detect import Platform

# Import config and enums from the config module
from utils.config import VLMChatConfig, CameraModel, Device

# Assuming BaseCamera is in a 'camera' subdirectory
from .camera_base import BaseCamera


logger = logging.getLogger(__name__)


class CameraFactory:
    """
    Factory for creating camera instances.

    This factory is instantiated with the application's configuration
    and runtime platform, and is used to create specific camera
    implementations.
    """

    def __init__(self):
        """
        Initialize the factory with the main application configuration.

        Args:
            config: The main VLMChatConfig object.
        """

    @classmethod
    def create_camera(
        cls,
        config: VLMChatConfig,
        collector: Collector,
        model: Optional[CameraModel] = None,
        device: Optional[Device] = None,
        args: Optional[dict[str, Any]] = None,
    ) -> BaseCamera:
        """
        Create a camera instance based on specifications.

        If model or device are not provided, the defaults from the
        injected CameraConfig will be used.

        Args:
            model: Camera model type (IMX500, IMX477, etc.).
            device: Device identifier (camera0, camera1).
            args: Optional arguments, primarily for ImageLibraryCamera.

        Returns:
            BaseCamera: Appropriate camera instance.

        Raises:
            ValueError: If unsupported model/platform combination is requested.
            NotImplementedError: If requested configuration is not yet implemented.
        """
       

        # Resolve model and device. Use provided args, fall back to config.
        model_to_use = model or config.camera.camera_model
        device_to_use = device or config.camera.camera_device

        logger.debug(
            f"CameraFactory: creating camera. "
            f"Model={model_to_use.value}, "
            f"Device={device_to_use.value}, "
            f"Platform={config.platform.value}"
        )

        # Dispatch based on resolved values
        if config.platform == Platform.RPI and model_to_use == CameraModel.IMX500:
            logger.info(f"CameraFactory: creating IMX500 camera on {config.platform.value} (device={device_to_use.value})")
            from ..utils.imx500_camera import IMX500Camera # Assumed path
            return IMX500Camera(platform=config.platform, device=device_to_use)

        elif model_to_use == CameraModel.IMX477:
            raise NotImplementedError(f"IMX477 camera support not yet implemented")

        elif config.platform == Platform.JETSON and model_to_use == CameraModel.IMX219:
            logger.info(f"CameraFactory: creating IMX219 camera on {config.platform.value} (device={device_to_use.value})")
            from ..utils.imx219_camera import IMX219Camera # Assumed path
            return IMX219Camera(platform=config.platform, device=device_to_use)

        elif model_to_use == CameraModel.IMAGE_LIBRARY:
            logger.info(f"CameraFactory: creating ImageLibrary camera")
            from .image_library_camera import ImageLibraryCamera # Assumed path
            return ImageLibraryCamera(
                config.camera.image_library_dir,
                config.camera.width,
                config.camera.height,
                config.camera.framerate,
                config.platform,
                device_to_use,
                config.paths.captured_images_dir,
                collector
            )

        else:
            logger.warning(
                f"CameraFactory: No specific camera found for "
                f"model={model_to_use.value} on platform={config.platform.value}. "
                f"Returning NoneCamera."
            )
            return NoneCamera(config, collector)

    def get_supported_models(self) -> list[CameraModel]:
        """Get list of supported camera models."""
        return [
            CameraModel.IMX500,
            CameraModel.IMX219,
            CameraModel.IMAGE_LIBRARY,
            CameraModel.NONE,
        ]

    def supports_detection(self, model: CameraModel) -> bool:
        """Check if a camera model supports object detection."""
        return model == CameraModel.IMX500