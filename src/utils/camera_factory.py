"""
Camera factory for creating camera instances.

This module provides a factory class for creating appropriate camera instances
based on model, platform, and device specifications.
"""

from typing import Union

from .camera_base import BaseCamera, CameraModel, Platform, Device
from .detection_base import ObjectDetectionInterface


class CameraFactory:
    """
    Factory class for creating camera instances.

    Provides static methods to create appropriate camera implementations
    based on hardware configuration and capability requirements.
    """

    @staticmethod
    def create_camera(
        model: CameraModel,
        platform: Platform = Platform.RPI,
        device: Device = Device.CAMERA0,
        with_detection: bool = False,
        args=None
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
        if platform == Platform.RPI and model == CameraModel.IMX500:
            if with_detection:
                from .imx500_detection import IMX500ObjectDetection
                
                return IMX500ObjectDetection(args=args, platform=platform, device=device)
            else:
                from .imx500_camera import IMX500Camera
                
                return IMX500Camera(platform=platform, device=device)

        elif model == CameraModel.IMX477:
            # TODO: Implement IMX477 camera support
            raise NotImplementedError(f"IMX477 camera support not yet implemented")

        elif platform == Platform.JETSON and model == CameraModel.IMX219:
            from .imx219_camera import IMX219Camera
                
            return IMX219Camera(platform=platform, device=device)

        else:
            raise ValueError(f"Unsupported camera model: {model} on platform: {platform}")

    @staticmethod
    def create_detection_camera(
        model: CameraModel,
        platform: Platform = Platform.RPI,
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
    def supports_detection(model: CameraModel) -> bool:
        """
        Check if a camera model supports object detection.

        Args:
            model: Camera model to check

        Returns:
            True if model supports detection, False otherwise
        """
        return model == CameraModel.IMX500
