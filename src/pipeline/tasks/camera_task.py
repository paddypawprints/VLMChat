"""
Camera task adapter for pipeline integration.

Wraps a camera instance and adapts it to the pipeline task interface.
"""

from typing import Optional, Dict
from PIL import Image

from ..task_base import BaseTask, Context, ContextDataType
from ...camera.camera_base import BaseCamera


class CameraTask(BaseTask):
    """
    Pipeline task adapter for camera capture.
    
    Can be initialized in two ways:
    1. With camera instance: CameraTask(camera, "cam0")
    2. Via configure(): CameraTask("cam0").configure({"type": "imx219", "device": "0"})
    
    Wraps a camera instance and captures single images, storing them in context.
    """
    
    def __init__(self, camera: Optional[BaseCamera] = None, task_id: str = "camera"):
        """
        Initialize camera task.
        
        Args:
            camera: Optional camera instance to wrap (can be configured later)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.camera = camera
        
        # Define contracts - camera produces an IMAGE
        self.input_contract = {}  # Source task, no inputs
        self.output_contract = {ContextDataType.IMAGE: Image.Image}
    
    def configure(self, params: Dict[str, str]) -> None:
        """
        Configure camera from parameters (DSL support).
        
        Args:
            params: Dictionary with camera configuration
                - type: Camera type (imx219, imx500, image_library, none)
                - device: Device identifier (e.g., "0" for /dev/video0)
                - resolution: Resolution string (e.g., "640x480")
                - Other camera-specific params
        
        Example:
            task.configure({"type": "imx219", "device": "0", "resolution": "640x480"})
        """
        if self.camera is not None:
            # Already have a camera, just pass through params
            return
        
        from ...camera.camera_factory import CameraFactory
        from ...utils.config import VLMChatConfig, CameraConfig, CameraModel, Device, Platform
        
        camera_type = params.get("type", "none")
        device_str = params.get("device", "0")
        
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
        
        # Create camera using factory
        self.camera = CameraFactory.create_camera(config, self.collector)
        
        if self.camera is None:
            raise ValueError(f"Failed to create camera with type '{camera_type}'")
    
    def run(self, context: Context) -> Context:
        """
        Capture image from camera and store in context.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with captured image
            
        Raises:
            RuntimeError: If camera is not configured
        """
        if self.camera is None:
            raise RuntimeError(f"Task {self.task_id}: Camera not configured. "
                             "Call configure() or pass camera to __init__")
        
        # Capture single image
        filepath, pil_image = self.camera.capture_single_image()
        
        # Store in context as a list (Context expects lists for all data types)
        if ContextDataType.IMAGE not in context.data:
            context.data[ContextDataType.IMAGE] = []
        context.data[ContextDataType.IMAGE].append(pil_image)
        
        # Optionally store filepath as metadata
        if not hasattr(context, 'metadata'):
            context.metadata = {}
        context.metadata['image_path'] = filepath
        
        return context
