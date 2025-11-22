"""
Object detector task adapter for pipeline integration.

Wraps a single ObjectDetector instance and adapts it to the pipeline task interface.
Each detector in a chain becomes its own pipeline task for proper metrics and control.
"""

from typing import Optional, List, Dict
import numpy as np
from PIL import Image

from .task_base import BaseTask, Context, ContextDataType, register_task
from ..object_detector.detection_base import ObjectDetector, Detection


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
    
    def __init__(self, detector: Optional[ObjectDetector] = None, task_id: str = "detector"):
        """
        Initialize detector task.
        
        Args:
            detector: Optional ObjectDetector instance (can be configured later)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.detector = detector
        
        # Define contracts
        # Input: Requires IMAGE and optionally DETECTIONS from previous stage
        # Output: Produces or updates DETECTIONS
        self.input_contract = {ContextDataType.IMAGE: Image.Image}
        self.output_contract = {ContextDataType.DETECTIONS: list}
    
    def configure(self, params: Dict[str, str]) -> None:
        """
        Configure detector from parameters (DSL support).
        
        Args:
            params: Dictionary with detector configuration
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
            task.configure({"type": "yolo", "model": "yolov8n.pt", "confidence": "0.5"})
            task.configure({"type": "clusterer", "max_clusters": "4", "merge_threshold": "1.5"})
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
            # Get source detector from params (required)
            if "source" not in params:
                raise ValueError("Clusterer requires 'source' detector in params")
            
            source_detector = params["source"]  # Should be an ObjectDetector instance
            
            # Get clusterer configuration from params or config
            max_clusters = int(params.get("max_clusters", "4"))
            merge_threshold = float(params.get("merge_threshold", "1.5"))
            proximity_weight = float(params.get("proximity_weight", "1.0"))
            size_weight = float(params.get("size_weight", "0.5"))
            semantic_pair_weight = float(params.get("semantic_pair_weight", "1.0"))
            semantic_single_weight = float(params.get("semantic_single_weight", "0.5"))
            
            # Get semantic provider from params (required for clusterer)
            if "semantic_provider" not in params:
                raise ValueError("Clusterer requires 'semantic_provider' in params")
            
            semantic_provider = params["semantic_provider"]
            
            # Create clusterer
            from ..object_detector.object_clusterer import ObjectClusterer
            self.detector = ObjectClusterer(
                source=source_detector,
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
        
        else:
            raise ValueError(f"Unknown detector type: {detector_type}. "
                           "Supported: yolo, yolo_cpu, imx500, clusterer")
    
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
        if self.detector is None:
            raise RuntimeError(f"Task {self.task_id}: Detector not configured. "
                             "Call configure() or pass detector to __init__")
        
        # Ensure detector is started (lazy initialization)
        if hasattr(self.detector, 'start') and hasattr(self.detector, '_ready'):
            if not self.detector._ready:
                self.detector.start()
        
        # Get image from context (stored as a list)
        image_list = context.data.get(ContextDataType.IMAGE)
        if not image_list:
            raise ValueError(f"Task {self.task_id}: IMAGE not found in context")
        
        # Get the first image (for now, we process single images)
        image = image_list[0] if isinstance(image_list, list) else image_list
        
        # Get existing detections (if any from previous stage)
        existing_detections = context.data.get(ContextDataType.DETECTIONS, [])
        
        # Run this detector stage
        # Pass existing detections so the detector can process/filter them
        detections = self.detector.detect(image, existing_detections)
        
        # Store updated detections in context
        context.data[ContextDataType.DETECTIONS] = detections
        
        return context
