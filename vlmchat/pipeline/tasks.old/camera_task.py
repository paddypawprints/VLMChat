"""
Camera task adapter for pipeline integration.

Wraps a camera instance and adapts it to the pipeline task interface.
"""

from typing import Optional, Dict, Any, Union
from PIL import Image

from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...camera.camera_base import BaseCamera


@register_task('camera')
class CameraTask(BaseTask):
    """
    Pipeline task adapter for camera capture.
    
    Can be initialized in two ways:
    1. With camera instance: CameraTask(camera, "cam0")
    2. Via configure(): CameraTask("cam0").configure({"type": "imx219", "device": "0"})
    
    Wraps a camera instance and captures single images, storing them in context.
    """
    
    def __init__(self, task_id: str = "camera", camera: Optional[BaseCamera] = None):
        """
        Initialize camera task.
        
        Args:
            task_id: Unique identifier for this task
            camera: Optional camera instance to wrap (can be configured later via configure())
        """
        super().__init__(task_id)
        self.camera = camera
        
        # Store configuration for deferred camera creation
        self._camera_config: Optional[Dict[str, Any]] = None
        
        # Define contracts - camera produces an IMAGE
        self.input_contract = {}  # Source task, no inputs
        self.output_contract = {ContextDataType.IMAGE: Image.Image}
        
        # Declare native format (cameras produce PIL images)
        from ..image_format import ImageFormat
        self.native_output_format = ImageFormat.PIL
    
    def configure(self, **params) -> None:
        """
        Configure camera from parameters (DSL support).
        
        Camera creation is deferred until run() so that collector is available.
        
        Args:
            **params: Keyword arguments with camera configuration
                - type: Camera type (imx219, imx500, image_library, none)
                - device: Device identifier (e.g., "0" for /dev/video0)
                - resolution: Resolution string (e.g., "640x480")
                - image_path: For type="none", path or URL to image file
                - Other camera-specific params
        
        Example:
            task.configure(type="none", image_path="https://example.com/image.jpg")
            task.configure(type="none", image_path="/path/to/image.jpg")
        """
        if self.camera is not None:
            # Already have a camera, just pass through params
            return
        
        # Store config for deferred camera creation in run()
        self._camera_config = params
    
    def _create_camera_from_config(self) -> None:
        """Create camera from stored configuration (called during run)."""
        if self.camera is not None or self._camera_config is None:
            return
        
        from ...camera.camera_factory import CameraFactory
        from ...utils.config import VLMChatConfig, CameraConfig, CameraModel, Device, Platform
        
        params = self._camera_config
        camera_type = params.get("type", "none")
        device_str = params.get("device", "0")
        image_path = params.get("image_path", None)  # For NoneCamera
        
        # Map type string to CameraModel enum
        camera_model_map = {
            "none": CameraModel.NONE,
            "imx219": CameraModel.IMX219,
            "imx500": CameraModel.IMX500,
            "image_library": CameraModel.IMAGE_LIBRARY,
        }
        camera_model = camera_model_map.get(camera_type.lower(), CameraModel.NONE)
        
        # Map device string to Device enum
        device_map = {
            "0": Device.CAMERA0,
            "1": Device.CAMERA1,
            "camera0": Device.CAMERA0,
            "camera1": Device.CAMERA1,
        }
        device = device_map.get(device_str.lower(), Device.CAMERA0)
        
        # Build CameraConfig
        camera_config = CameraConfig(
            camera_model=camera_model,
            camera_device=device
        )
        
        # Parse resolution if provided
        if "resolution" in params:
            try:
                width, height = params["resolution"].split("x")
                camera_config.width = int(width)
                camera_config.height = int(height)
            except (ValueError, AttributeError):
                pass
        
        # Build minimal VLMChatConfig with camera config
        config = VLMChatConfig(camera=camera_config)
        
        # Create camera using factory (collector is now available)
        if camera_model == CameraModel.NONE and image_path:
            # Special case: NoneCamera with custom image source
            from ...camera.none_camera import NoneCamera
            self.camera = NoneCamera(config, self.collector, image_source=image_path)  # type: ignore[arg-type]
        else:
            self.camera = CameraFactory.create_camera(config, self.collector)  # type: ignore[arg-type]
        
        if self.camera is None:
            raise ValueError(f"Failed to create camera with type '{camera_type}'")
    
    def run(self, context: Context) -> Context:
        """
        Capture image from camera and store in context and environment.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with captured image
            
        Raises:
            RuntimeError: If camera is not configured
        """
        # Create camera if needed (deferred from configure())
        if self.camera is None and self._camera_config is not None:
            self._create_camera_from_config()
        
        if self.camera is None:
            raise RuntimeError(f"Task {self.task_id}: Camera not configured. "
                             "Call configure() or pass camera to __init__")
        
        # Capture single image
        filepath, pil_image = self.camera.capture_single_image()
        
        # Store in context as a list (Context expects lists for all data types)
        if ContextDataType.IMAGE not in context.data:
            context.data[ContextDataType.IMAGE] = []
        context.data[ContextDataType.IMAGE].append(pil_image)
        
        # Store in environment using helper method (for chat app and other tasks to access)
        self.env_set("current_image", pil_image)
        self.env_set("image_path", filepath)
        
        return context
    
    def describe(self) -> str:
        """Return description of what this task does."""
        return "Captures an image from a camera device and stores it in the pipeline context and environment."
    
    def describe_parameters(self) -> Dict[str, Union[str, Dict[str, Any]]]:
        """Return parameter descriptions for camera configuration."""
        return {
            "type": {
                "description": "Camera device type",
                "type": "str",
                "choices": ["none", "imx219", "imx500", "image_library"],
                "default": "none",
                "example": "imx219"
            },
            "device": {
                "description": "Device identifier for camera",
                "type": "str",
                "choices": ["0", "1", "camera0", "camera1"],
                "default": "0",
                "example": "0"
            },
            "resolution": {
                "description": "Image resolution in WIDTHxHEIGHT format",
                "type": "str",
                "format": "WIDTHxHEIGHT (e.g., 640x480, 1920x1080)",
                "example": "640x480"
            }
        }
