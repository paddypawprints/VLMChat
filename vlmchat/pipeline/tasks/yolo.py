"""
YOLO object detection tasks.

Simple task wrappers around YOLO models. Each model gets its own task.
"""

import logging
from typing import Optional
from pathlib import Path

from ..core.task_base import BaseTask, Context, ContextDataType, register_task
from ..detection import Detection
from ..image.formats import ImageFormat
from ..models.yolo_ultralytics import YoloUltralytics
from ..models.yolo_tensorrt import YoloTensorRT

logger = logging.getLogger(__name__)


@register_task('yolo_ultralytics')
class YoloUltralyticsTask(BaseTask):
    """
    YOLO object detection using Ultralytics backend (CPU inference).
    
    Consumes images from input label and produces Detection objects on output label.
    Input images are removed from context after processing (consume-by-default).
    
    Contract:
        Input: IMAGE[input_label] - ImageContainer objects to detect from
        Output: IMAGE[output_label] - Detection objects (extends ImageContainer)
    
    Usage:
        # Basic: frame -> detections
        yolo_ultralytics()
        
        # Custom labels
        yolo_ultralytics(input=frame, output=detections)
        
        # With parameters
        yolo_ultralytics(model=yolov8n.pt, confidence=0.5, iou=0.45)
    """
    
    def __init__(self,
                 task_id: str = "yolo_ultralytics",
                 model_path: Optional[str] = None,
                 confidence: float = 0.25,
                 iou: float = 0.45,
                 input_label: str = "frame",
                 output_label: str = "detections"):
        """
        Initialize YOLO Ultralytics task.
        
        Args:
            task_id: Unique task identifier
            model_path: Path to YOLO model file (.pt). None = use config default
            confidence: Minimum confidence threshold (0.0 to 1.0)
            iou: IoU threshold for NMS (0.0 to 1.0)
            input_label: Label to read images from (default: "frame")
            output_label: Label to write detections to (default: "detections")
        """
        super().__init__(task_id)
        
        # Parameters
        self.model_path = Path(model_path).expanduser() if model_path else None
        self.confidence = confidence
        self.iou = iou
        self.input_label = input_label
        self.output_label = output_label
        
        # Model (initialized on first run)
        self.model: Optional[YoloUltralytics] = None
        
        # Declare contracts with format requirements
        from ...cache.types import CachedItemType
        self.input_contract = {
            ContextDataType.IMAGE: {
                input_label: (CachedItemType.IMAGE, "numpy")  # YOLO needs numpy arrays
            }
        }
        self.output_contract = {
            ContextDataType.IMAGE: {
                output_label: (CachedItemType.IMAGE, None)  # Detections (virtual images)
            }
        }
    
    def configure(self, **kwargs) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            **kwargs: Configuration parameters
                - model: Model path (str)
                - confidence: Confidence threshold (float)
                - iou: IoU threshold (float)
                - input: Input label (str)
                - output: Output label (str)
        """
        if "model" in kwargs:
            self.model_path = Path(kwargs["model"]).expanduser()
        
        if "confidence" in kwargs:
            try:
                self.confidence = float(kwargs["confidence"])
            except ValueError:
                logger.warning(f"Invalid confidence value: {kwargs['confidence']}")
        
        if "iou" in kwargs:
            try:
                self.iou = float(kwargs["iou"])
            except ValueError:
                logger.warning(f"Invalid iou value: {kwargs['iou']}")
        
        if "input" in kwargs:
            self.input_label = kwargs["input"]
            # Update contract with numpy format
            from ...cache.types import CachedItemType
            self.input_contract = {
                ContextDataType.IMAGE: {
                    self.input_label: (CachedItemType.IMAGE, "numpy")
                }
            }
        
        if "output" in kwargs:
            self.output_label = kwargs["output"]
            # Update contract
            from ...cache.types import CachedItemType
            self.output_contract = {
                ContextDataType.IMAGE: {
                    self.output_label: (CachedItemType.IMAGE, None)
                }
            }
    
    def _ensure_model(self) -> None:
        """Initialize model on first use."""
        if self.model is not None:
            return
        
        # Use model_path if specified, otherwise default
        model_path = self.model_path if self.model_path else "~/yolov8n.pt"
        
        # Initialize model
        self.model = YoloUltralytics(model_path=str(model_path))
        
        logger.info(f"YOLO Ultralytics model initialized: {model_path}")
    
    def run(self, context: Context) -> Context:
        """
        Run YOLO detection on input images.
        
        Args:
            context: Pipeline context with IMAGE[input_label] items
            
        Returns:
            Context with IMAGE[output_label] Detection objects added,
            input images removed (consume-by-default)
        """
        # Initialize model on first run
        self._ensure_model()
        
        # Get input images
        frames = context.get_data(ContextDataType.IMAGE, self.input_label)
        if not frames:
            logger.debug(f"No images in label '{self.input_label}'")
            return context
        
        # Consume input (remove from context)
        context.clear(data_type=ContextDataType.IMAGE, label=self.input_label)
        
        # Process each frame
        for cached_item in frames:
            frame = cached_item.get_cached_item()  # ImageContainer
            
            # Run detection - model returns Detection objects directly
            detections = self.model.detect(
                frame,
                confidence=self.confidence,
                iou=self.iou
            )
            
            # Add detections to context
            for detection in detections:
                context.add_data(ContextDataType.IMAGE, detection, self.output_label)
            
            logger.debug(f"YOLO detected {len(detections)} objects")
        
        return context
    
    def __str__(self) -> str:
        """String representation."""
        return f"YoloUltralyticsTask(input={self.input_label}, output={self.output_label}, conf={self.confidence})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()


@register_task('yolo_tensorrt')
class YoloTensorRTTask(BaseTask):
    """
    YOLO object detection using TensorRT backend (GPU inference).
    
    Consumes images from input label and produces Detection objects on output label.
    Input images are removed from context after processing (consume-by-default).
    
    Contract:
        Input: IMAGE[input_label] - ImageContainer objects to detect from
        Output: IMAGE[output_label] - Detection objects (extends ImageContainer)
    
    Usage:
        # Basic: frame -> detections
        yolo_tensorrt()
        
        # Custom labels
        yolo_tensorrt(input=frame, output=detections)
        
        # With parameters
        yolo_tensorrt(engine=yolov8n.engine, confidence=0.5, iou=0.45)
    """
    
    def __init__(self,
                 task_id: str = "yolo_tensorrt",
                 engine_path: Optional[str] = None,
                 confidence: float = 0.25,
                 iou: float = 0.45,
                 input_label: str = "frame",
                 output_label: str = "detections"):
        """
        Initialize YOLO TensorRT task.
        
        Args:
            task_id: Unique task identifier
            engine_path: Path to TensorRT engine file (.engine). None = use config default
            confidence: Minimum confidence threshold (0.0 to 1.0)
            iou: IoU threshold for NMS (0.0 to 1.0)
            input_label: Label to read images from (default: "frame")
            output_label: Label to write detections to (default: "detections")
        """
        super().__init__(task_id)
        
        # Parameters
        self.engine_path = Path(engine_path).expanduser() if engine_path else None
        self.confidence = confidence
        self.iou = iou
        self.input_label = input_label
        self.output_label = output_label
        
        # Backend (initialized on first run)
        self.model: Optional[YoloTensorRT] = None
        
        # Declare contracts with format requirements
        from ...cache.types import CachedItemType
        self.input_contract = {
            ContextDataType.IMAGE: {
                input_label: (CachedItemType.IMAGE, "numpy")  # YOLO needs numpy arrays
            }
        }
        self.output_contract = {
            ContextDataType.IMAGE: {
                output_label: (CachedItemType.IMAGE, None)  # Detections (virtual images)
            }
        }
    
    def configure(self, **kwargs) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            **kwargs: Configuration parameters
                - engine: Engine path (str)
                - confidence: Confidence threshold (float)
                - iou: IoU threshold (float)
                - input: Input label (str)
                - output: Output label (str)
        """
        if "engine" in kwargs:
            self.engine_path = Path(kwargs["engine"]).expanduser()
        
        if "confidence" in kwargs:
            try:
                self.confidence = float(kwargs["confidence"])
            except ValueError:
                logger.warning(f"Invalid confidence value: {kwargs['confidence']}")
        
        if "iou" in kwargs:
            try:
                self.iou = float(kwargs["iou"])
            except ValueError:
                logger.warning(f"Invalid iou value: {kwargs['iou']}")
        
        if "input" in kwargs:
            self.input_label = kwargs["input"]
            # Update contract with numpy format
            from ...cache.types import CachedItemType
            self.input_contract = {
                ContextDataType.IMAGE: {
                    self.input_label: (CachedItemType.IMAGE, "numpy")
                }
            }
        
        if "output" in kwargs:
            self.output_label = kwargs["output"]
            # Update contract
            from ...cache.types import CachedItemType
            self.output_contract = {
                ContextDataType.IMAGE: {
                    self.output_label: (CachedItemType.IMAGE, None)
                }
            }
    
    def _ensure_model(self, context: Context) -> None:
        """Initialize TensorRT model on first run."""
        if self.model is not None:
            return
        
        # Determine engine path
        if self.engine_path is None:
            # Try to get from config
            config = context.config
            if config and hasattr(config, 'model') and hasattr(config.model, 'yolo_engine_path'):
                self.engine_path = Path(config.model.yolo_engine_path).expanduser()
            else:
                raise RuntimeError("No TensorRT engine path specified")
        
        # Initialize model
        self.model = YoloTensorRT(engine_path=str(self.engine_path))
        
        logger.info(f"YOLO TensorRT model initialized: {self.engine_path}")
    
    def run(self, context: Context) -> Context:
        """
        Run YOLO detection on input images using TensorRT.
        
        Args:
            context: Pipeline context with IMAGE[input_label] items
            
        Returns:
            Context with IMAGE[output_label] Detection objects added,
            input images removed (consume-by-default)
        """
        # Initialize model on first run
        self._ensure_model(context)
        
        # Get input images
        frames = context.get_data(ContextDataType.IMAGE, self.input_label)
        if not frames:
            logger.debug(f"No images in label '{self.input_label}'")
            return context
        
        # Consume input (remove from context)
        context.clear(data_type=ContextDataType.IMAGE, label=self.input_label)
        
        # Process each frame
        for cached_item in frames:
            frame = cached_item.get_cached_item()  # ImageContainer
            
            # Run TensorRT detection
            detections = self.model.detect(frame, confidence=self.confidence, iou=self.iou)
            
            # Add detections to context
            for detection in detections:
                context.add_data(ContextDataType.IMAGE, detection, self.output_label)
            
            logger.debug(f"YOLO TensorRT detected {len(detections)} objects")
        
        return context
    
    def __str__(self) -> str:
        """String representation."""
        return f"YoloTensorRTTask(input={self.input_label}, output={self.output_label}, conf={self.confidence})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()
