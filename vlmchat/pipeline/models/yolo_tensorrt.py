"""
YOLO TensorRT backend for GPU-accelerated inference.

Self-contained implementation for TensorRT-optimized YOLO models.
Returns Detection objects ready for pipeline use.
"""

import logging
import numpy as np
from typing import List, Optional
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    cuda = None

from ..detection import Detection
from ..cache.image import ImageContainer
from ..image.formats import ImageFormat
from ..categories import CocoCategory

logger = logging.getLogger(__name__)


class YoloTensorRT:
    """
    TensorRT-optimized YOLO backend for GPU inference.
    
    Provides high-performance object detection on Jetson/GPU devices.
    """
    
    def __init__(
        self, 
        engine_path: str,
        class_names: Optional[List[str]] = None
    ):
        """
        Initialize TensorRT YOLO model.
        
        Args:
            engine_path: Path to TensorRT engine file (.engine or .trt)
            class_names: List of class names (defaults to COCO-80)
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT not available. Install with: "
                "pip install tensorrt pycuda"
            )
        
        self.engine_path = Path(engine_path).expanduser()
        
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found at: {self.engine_path}")
        
        # Load engine
        logger.info(f"Loading TensorRT engine from: {self.engine_path}")
        self.engine, self.context = self._load_engine()
        
        # Setup input/output bindings
        self._setup_bindings()
        
        # Class names (default to COCO-80)
        if class_names is None:
            # Use standard COCO class names
            self.class_names = [
                'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
                'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
                'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                'scissors', 'teddy bear', 'hair drier', 'toothbrush'
            ]
        else:
            self.class_names = class_names
        
        logger.info(f"TensorRT YOLO loaded: {len(self.class_names)} classes")
    
    def __del__(self):
        """Cleanup TensorRT resources and CUDA memory."""
        try:
            # Free CUDA device memory for inputs
            if hasattr(self, 'inputs'):
                for inp in self.inputs:
                    if 'device' in inp and inp['device']:
                        inp['device'].free()
            
            # Free CUDA device memory for outputs
            if hasattr(self, 'outputs'):
                for out in self.outputs:
                    if 'device' in out and out['device']:
                        out['device'].free()
            
            # Destroy TensorRT context
            if hasattr(self, 'context') and self.context:
                del self.context
            
            # Delete engine
            if hasattr(self, 'engine') and self.engine:
                del self.engine
        except Exception:
            pass  # Ignore cleanup errors
    
    def _load_engine(self):
        """Load TensorRT engine from file."""
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        
        return engine, context
    
    def _setup_bindings(self):
        """Setup input/output bindings for inference."""
        # Allocate buffers
        self.bindings = []
        self.inputs = []
        self.outputs = []
        
        # Use newer TensorRT 10+ API
        num_io_tensors = self.engine.num_io_tensors
        
        for i in range(num_io_tensors):
            tensor_name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(tensor_name)
            shape = self.engine.get_tensor_shape(tensor_name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
            
            # Allocate host and device buffers
            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': tensor_name,
                    'shape': shape,
                    'dtype': dtype,
                    'host': host_mem,
                    'device': device_mem
                })
            else:  # OUTPUT
                self.outputs.append({
                    'name': tensor_name,
                    'shape': shape,
                    'dtype': dtype,
                    'host': host_mem,
                    'device': device_mem
                })
        
        logger.debug(f"Input: {self.inputs[0]['name']} {self.inputs[0]['shape']}")
        logger.debug(f"Output: {self.outputs[0]['name']} {self.outputs[0]['shape']}")
        
        # Log shapes for debugging
        logger.info(f"TensorRT engine loaded: input={self.inputs[0]['shape']}, output={self.outputs[0]['shape']}")
    
    @property
    def is_available(self) -> bool:
        """Check if model is loaded and ready."""
        return self.engine is not None and self.context is not None
    
    @property
    def preferred_format(self) -> ImageFormat:
        """Preferred input format (numpy BGR)."""
        return ImageFormat.NUMPY
    
    def _preprocess(self, image: np.ndarray) -> tuple:
        """
        Preprocess image for YOLO inference.
        
        Args:
            image: Input image (HWC, BGR, uint8)
            
        Returns:
            Tuple of (preprocessed, scale, pad_info)
        """
        # Get input shape
        input_shape = self.inputs[0]['shape']
        batch_size, channels, height, width = input_shape
        
        # Original dimensions
        orig_h, orig_w = image.shape[:2]
        
        # Calculate resize ratio (letterbox)
        r = min(height / orig_h, width / orig_w)
        new_h, new_w = int(orig_h * r), int(orig_w * r)
        
        # Resize
        import cv2
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to input size (letterbox)
        dh, dw = height - new_h, width - new_w
        top, left = dh // 2, dw // 2
        bottom, right = dh - top, dw - left
        
        padded = cv2.copyMakeBorder(
            resized, top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        
        # BGR to RGB, HWC to CHW
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        chw = rgb.transpose(2, 0, 1)
        
        # Normalize to [0, 1] and add batch dimension
        normalized = chw.astype(np.float32) / 255.0
        batched = np.expand_dims(normalized, axis=0)
        
        # Metadata for post-processing
        meta = {
            'ratio': r,
            'pad': (top, left),
            'orig_shape': (orig_h, orig_w)
        }
        
        return batched, r, meta
    
    def _postprocess(
        self, 
        output: np.ndarray,
        scale: float,
        meta: dict,
        confidence: float,
        iou: float
    ) -> List[tuple]:
        """
        Post-process YOLO output to detections.
        
        Args:
            output: Raw model output
            scale: Scale factor used in preprocessing
            meta: Preprocessing metadata
            confidence: Confidence threshold
            iou: IoU threshold for NMS
            
        Returns:
            List of (bbox, conf, class_id) tuples
        """
        # YOLOv8 output format can be:
        # - [batch, 84, num_boxes] - transposed format (cx, cy, w, h + 80 classes)
        # - [batch, num_boxes, 84] - standard format
        
        logger.debug(f"Raw output shape: {output.shape}")
        
        # Handle different output formats
        if len(output.shape) == 3:
            if output.shape[1] == 84:
                # Format: [batch, 84, num_boxes] - need to transpose
                predictions = output[0].T  # Now [num_boxes, 84]
            elif output.shape[2] == 84:
                # Format: [batch, num_boxes, 84]
                predictions = output[0]
            else:
                logger.error(f"Unexpected output shape: {output.shape}")
                return []
        else:
            logger.error(f"Unexpected output dimensions: {output.shape}")
            return []
        
        logger.debug(f"Predictions shape after reshape: {predictions.shape}")
        
        # Extract boxes and scores
        # Format: [cx, cy, w, h, class0_score, class1_score, ..., class79_score]
        boxes = predictions[:, :4]  # cx, cy, w, h
        class_scores = predictions[:, 4:]  # 80 class scores
        
        # Get max class score and ID for each box
        max_scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)
        
        # Filter by confidence
        mask = max_scores >= confidence
        
        if not np.any(mask):
            return []
        
        boxes = boxes[mask]
        scores = max_scores[mask]
        class_ids = class_ids[mask]
        
        logger.debug(f"After confidence filter: {len(boxes)} detections")
        
        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # Adjust for preprocessing (remove padding and scale)
        top, left = meta['pad']
        ratio = meta['ratio']
        
        x1 = (x1 - left) / ratio
        y1 = (y1 - top) / ratio
        x2 = (x2 - left) / ratio
        y2 = (y2 - top) / ratio
        
        # Clip to image bounds
        orig_h, orig_w = meta['orig_shape']
        x1 = np.clip(x1, 0, orig_w)
        x2 = np.clip(x2, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        y2 = np.clip(y2, 0, orig_h)
        
        # Apply NMS
        import cv2
        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes_xyxy.tolist(),
            scores.tolist(),
            confidence,
            iou
        )
        
        if len(indices) == 0:
            return []
        
        # Build result list
        results = []
        for i in indices.flatten():
            results.append((
                (float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
                float(scores[i]),
                int(class_ids[i])
            ))
        
        return results
    
    def detect(
        self,
        image: ImageContainer,
        confidence: float = 0.25,
        iou: float = 0.45
    ) -> List[Detection]:
        """
        Run YOLO detection on an image.
        
        Args:
            image: ImageContainer to detect from
            confidence: Minimum confidence threshold (0.0 to 1.0)
            iou: IoU threshold for NMS (0.0 to 1.0)
            
        Returns:
            List of Detection objects
        """
        if not self.is_available:
            raise RuntimeError("TensorRT engine not loaded")
        
        # Get image as numpy array (BGR)
        numpy_image = image.get(ImageFormat.NUMPY)
        
        # Preprocess
        preprocessed, scale, meta = self._preprocess(numpy_image)
        
        # Copy input to device
        np.copyto(self.inputs[0]['host'], preprocessed.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        
        # Run inference
        self.context.execute_v2(self.bindings)
        
        # Copy output from device
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        output = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])
        
        # Post-process to detections
        detections_raw = self._postprocess(output, scale, meta, confidence, iou)
        
        # Convert to Detection objects
        detections = []
        for bbox, conf, class_id in detections_raw:
            # Get category
            category = CocoCategory.from_id(class_id)
            if category is None:
                logger.warning(f"Unknown COCO class ID: {class_id}, skipping detection")
                continue
            
            detection = Detection(
                bbox=bbox,
                confidence=conf,
                category=category,
                source_image=image
            )
            detections.append(detection)
        
        logger.debug(f"TensorRT YOLO detected {len(detections)} objects")
        return detections
    
    def __str__(self) -> str:
        """String representation."""
        return f"YoloTensorRT(engine={self.engine_path.name}, classes={len(self.class_names)})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()
