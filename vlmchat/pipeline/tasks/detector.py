"""
Object detector task adapter for pipeline integration.

Wraps a single ObjectDetector instance and adapts it to the pipeline task interface.
Each detector in a chain becomes its own pipeline task for proper metrics and control.
"""

from typing import Optional, List, Dict
import numpy as np
from PIL import Image
import logging

from ..core.task_base import BaseTask, Context, ContextDataType, register_task
from ...object_detector.detection_base import ObjectDetector, Detection

logger = logging.getLogger(__name__)


@register_task('detect')
@register_task('detector')
class DetectorTask(BaseTask):
    """
    Pipeline task adapter for object detection.
    
    Can be initialized in two ways:
    1. With detector instance: DetectorTask(detector, "yolo")
    2. Via configure(): DetectorTask("yolo").configure({"model": "yolov8n.pt", "confidence": "0.5"})
    
    Wraps a SINGLE ObjectDetector instance (not a chain). Multiple detectors
    should be added as separate tasks in the pipeline to get proper metrics
    and control flow.
    
    Example:
        yolo_task = DetectorTask(YoloV8Detector(), "yolo")
        filter_task = DetectorTask(PersonFilter(), "person_filter")
        viewer_task = DetectorTask(DetectionViewer(viewer=viewer), "viewer")
        
        pipeline.add_task(yolo_task)
        pipeline.add_task(filter_task)
        pipeline.add_task(viewer_task)
        pipeline.add_edge(yolo_task, filter_task)
        pipeline.add_edge(filter_task, viewer_task)
    """
    
    def __init__(self, detector=None, task_id: str = "detector", image_format: str = "pil"):
        """
        Initialize detector task.
        
        Args:
            detector: Optional ObjectDetector instance (can be configured later)
            task_id: Unique identifier for this task
            image_format: Expected image format (default: "pil"). Currently only "pil" supported.
        """
        super().__init__(task_id)
        self.detector = detector
        self.image_format = image_format.lower()
        
        # Validate image format
        if self.image_format != "pil":
            raise ValueError(f"Unsupported image format: {self.image_format}. Only 'pil' is currently supported.")
        
        # Define contracts
        # Input: Requires IMAGE and optionally DETECTIONS from previous stage
        # Output: Produces or updates DETECTIONS
        self.input_contract = {ContextDataType.IMAGE: Image.Image}
        self.output_contract = {ContextDataType.DETECTIONS: list}
        
        # Declare native format (detectors work with PIL images)
        from .image_format import ImageFormat
        self.native_input_format = ImageFormat.PIL
    
    def configure(self, **params) -> None:
        """
        Configure detector from parameters (DSL support).
        
        Args:
            **params: Keyword arguments with detector configuration
                - type: Detector type (yolo, yolo_cpu, imx500, clusterer, viewer)
                - model: Model path (e.g., "yolov8n.pt")
                - confidence: Confidence threshold (e.g., "0.5")
                - device: Device for inference (e.g., "cuda:0", "cpu")
                - max_clusters: Maximum number of clusters (clusterer only)
                - merge_threshold: Merge threshold (clusterer only)
                - proximity_weight: Weight for proximity cost (clusterer only)
                - size_weight: Weight for size cost (clusterer only)
                - semantic_pair_weight: Weight for semantic pair cost (clusterer only)
                - semantic_single_weight: Weight for semantic single cost (clusterer only)
        
        Example:
            task.configure(type="yolo", model="yolov8n.pt", confidence="0.5")
            task.configure(type="clusterer", max_clusters="4", merge_threshold="1.5")
        """
        if self.detector is not None:
            # Already have a detector
            return
        
        detector_type = params.get("type", "yolo_cpu")
        
        # Create appropriate detector based on type
        if detector_type == "yolo":
            from ..object_detector.yolo_detector import YoloV8Detector
            model_path = params.get("model", "yolov8n.pt")
            confidence = float(params.get("confidence", "0.25"))
            device = params.get("device", "cuda:0")
            self.detector = YoloV8Detector(
                model_path=model_path,
                confidence_threshold=confidence,
                device=device
            )
        
        elif detector_type == "yolo_cpu":
            from ..object_detector.yolo_detector_cpu import YoloV8Detector
            model_name = params.get("model", "yolov8n.pt")
            # Note: YoloV8Detector (CPU version) doesn't have confidence threshold in constructor
            # Don't start here - will be started later when wrapped or when first used
            self.detector = YoloV8Detector(model_name=model_name)
        
        elif detector_type == "imx500":
            from ..object_detector.imx500_detection import IMX500Detection
            self.detector = IMX500Detection()
        
        elif detector_type == "clusterer":
            # Clusterer will be initialized lazily in run() when CLIP model is available
            # Store params for later initialization
            self._clusterer_params = params
            self.detector = None  # Will be created in run()
        
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. "
                           "Supported: yolo, yolo_cpu, imx500, clusterer")
    
    def _initialize_clusterer(self, context: Context) -> None:
        """
        Lazy initialization of clusterer when CLIP model becomes available.
        Called from run() on first execution.
        """
        params = self._clusterer_params
        
        # Get clusterer configuration from params
        max_clusters = int(params.get("max_clusters", "4"))
        merge_threshold = float(params.get("merge_threshold", "1.5"))
        proximity_weight = float(params.get("proximity_weight", "1.0"))
        size_weight = float(params.get("size_weight", "0.5"))
        semantic_pair_weight = float(params.get("semantic_pair_weight", "1.0"))
        semantic_single_weight = float(params.get("semantic_single_weight", "0.5"))
        
        # Get semantic provider from environment or create one
        from ..pipeline_environment import Environment
        env = Environment.get_instance()
        
        semantic_provider = env.get("services", "clip", "semantic_provider")
        if semantic_provider is None:
            # Create a basic semantic provider with CLIP text model
            from ..object_detector.semantic_provider import ClipSemanticProvider
            from ..models.MobileClip.clip_text_model import ClipTextModel
            from ..utils.config import VLMChatConfig
            
            # Get or create CLIP text model
            text_model = env.get("services", "clip", "text_model")
            if text_model is None:
                # Initialize CLIP text model
                config = getattr(context, 'config', None) or VLMChatConfig()
                text_model = ClipTextModel(config=config, collector=None)
                env.set("services", "clip", "text_model", text_model)
            
            # Initialize with empty prompts (will be updated from TEXT context)
            semantic_provider = ClipSemanticProvider(
                text_model=text_model,
                user_prompts=[],
                embeddings_cache_path="category_pair_embeddings.json",
                batch_size=5
            )
            semantic_provider.start()
            
            # Store in environment for reuse
            env.set("services", "clip", "semantic_provider", semantic_provider)
        
        # Create clusterer - it will get detections from pipeline context, not from a source detector
        # ObjectClusterer requires a source in __init__, but we'll pass detections via detect() method
        from ..object_detector.object_clusterer import ObjectClusterer
        
        # Create a minimal dummy detector that just returns what it's given
        class _ContextDetector:
            """Dummy detector that returns detections from context."""
            def detect(self, image, detections):
                return detections
            def start(self, audit=False):
                pass
            def stop(self):
                pass
            def readiness(self):
                return True
            @property
            def _ready(self):
                return True
        
        self.detector = ObjectClusterer(
            source=_ContextDetector(),
            semantic_provider=semantic_provider,
            max_clusters=max_clusters,
            merge_threshold=merge_threshold,
            proximity_weight=proximity_weight,
            size_weight=size_weight,
            semantic_weights={
                "pair": semantic_pair_weight,
                "single": semantic_single_weight
            }
        )
        logger.info(f"[{self.task_id}] Clusterer initialized with max_clusters={max_clusters}")
    
    def run(self, context: Context) -> Context:
        """
        Run this detector stage on the image and existing detections in context.
        
        Args:
            context: Pipeline context containing IMAGE and optionally DETECTIONS
            
        Returns:
            Updated context with DETECTIONS added or modified
            
        Raises:
            RuntimeError: If detector is not configured
        """
        # Lazy initialization for clusterer (needs CLIP model from environment)
        if self.detector is None and hasattr(self, '_clusterer_params'):
            self._initialize_clusterer(context)
        
        if self.detector is None:
            raise RuntimeError(f"Task {self.task_id}: Detector not configured. "
                             "Call configure() or pass detector to __init__")
        
        # Ensure detector is started (lazy initialization)
        if hasattr(self.detector, 'start') and hasattr(self.detector, '_ready'):
            if not self.detector._ready:
                # Enable audit for clusterers
                audit_enabled = hasattr(self.detector, 'get_last_audit_log')
                self.detector.start(audit=audit_enabled)
        
        # Get image from context (stored as a list, may be ImageContainer or raw image)
        from .image_container import ImageContainer
        from .image_format import ImageFormat
        image_list = context.data.get(ContextDataType.IMAGE)
        if not image_list:
            raise ValueError(f"Task {self.task_id}: IMAGE not found in context")
        
        # Get the first image (for now, we process single images)
        image_item = image_list[0] if isinstance(image_list, list) else image_list
        
        # Extract from ImageContainer if needed
        if isinstance(image_item, ImageContainer) and self.native_input_format:
            image = image_item.get_format(self.native_input_format)
            if image is None:
                # Fallback to any available format
                for fmt in image_item.get_cached_formats():
                    image = image_item.get_format(fmt)
                    if image is not None:
                        break
        else:
            image = image_item
        
        # Get existing detections (if any from previous stage)
        existing_detections = context.data.get(ContextDataType.DETECTIONS, [])
        
        # If this is a clusterer and TEXT is available, update semantic prompts
        text_data = context.data.get(ContextDataType.TEXT)
        if text_data and hasattr(self.detector, 'set_text_prompts'):
            # TEXT is stored as a list of strings
            prompts = text_data if isinstance(text_data, list) else [text_data]
            self.detector.set_text_prompts(prompts)
        
        # Run this detector stage
        # Pass existing detections so the detector can process/filter them
        detections = self.detector.detect(image, existing_detections)
        
        # Store updated detections in context
        context.data[ContextDataType.DETECTIONS] = detections
        
        # If this is a clusterer, store audit log in context
        if hasattr(self.detector, 'get_last_audit_log'):
            audit_log = self.detector.get_last_audit_log()
            context.data[ContextDataType.AUDIT] = [str(audit_log)]
        
        return context
