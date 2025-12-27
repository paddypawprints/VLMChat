"""
TensorRT backend implementation for YOLO.

Provides YOLO runtime using TensorRT for optimized inference.
"""

import logging
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Any, Tuple
from pathlib import Path

# --- Third-party imports ---
try:
    import tensorrt as trt  # type: ignore[import-untyped]
    import pycuda.driver as cuda  # type: ignore[import-untyped]
    import pycuda.autoinit  # type: ignore[import-untyped]
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None  # type: ignore[assignment]
    cuda = None  # type: ignore[assignment]
# --- End third-party imports ---

from models.Yolov8n.runtime_base import YoloRuntimeBase, Image
from utils.config import VLMChatConfig

logger = logging.getLogger(__name__)


class TensorRTBackend(YoloRuntimeBase):
    """
    YOLO runtime implementation using TensorRT for optimized inference.
    """
    
    def __init__(self, config: VLMChatConfig):
        super().__init__(config)
        self.engine_path = config.model.yolo_engine_path
        self.model_size = 640
        self.input_name = "images"
        self.output_name = "output0"
        self.engine = None
        self.ctx = None
        self._is_ready = False
        
        # Load COCO class names
        self.class_names = self._load_class_names()
        
        if not TENSORRT_AVAILABLE:
            logger.error("TensorRTBackend Error: TensorRT or PyCUDA not installed.")
            return
        
        if not self.engine_path.exists():
            logger.error(f"YOLO TensorRT engine not found at: {self.engine_path}")
            return

        try:
            logger.info(f"Loading YOLO TensorRT engine from: {self.engine_path}")
            trt_logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[union-attr]
            runtime = trt.Runtime(trt_logger)  # type: ignore[union-attr]
            
            with open(self.engine_path, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                raise RuntimeError(f"Failed to deserialize engine from {self.engine_path}")
            
            self.ctx = self.engine.create_execution_context()
            self._is_ready = True
            logger.info("TensorRTBackend loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load YOLO TensorRT engine: {e}", exc_info=True)
            self._is_ready = False

    @property
    def native_image_format(self) -> str:
        """YOLO TensorRT backend expects NumPy arrays (BGR format)."""
        return "numpy"

    def _load_class_names(self) -> List[str]:
        """Load COCO class names from config file or use defaults."""
        import json
        
        detector_dir = Path(__file__).parent.parent.parent / "object_detector"
        coco_names_path = detector_dir / "coco_names.json"
        
        if coco_names_path.exists():
            try:
                with open(coco_names_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load COCO names from {coco_names_path}: {e}")
        
        # Fallback to numeric labels
        return [str(i) for i in range(80)]

    @property
    def is_available(self) -> bool:
        """Returns True if the engine loaded successfully."""
        return self._is_ready and self.engine is not None and self.ctx is not None

    def prepare_image(self, image: Image) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Prepares image using tensor->blob (stretch resize on CPU).
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            Tuple of (blob, scale, metadata)
        """
        h, w = image.shape[:2]
        arr = image.astype(np.float32) / 255.0
        device = torch.device("cpu")
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
        tensor_resized = F.interpolate(
            tensor,
            size=(self.model_size, self.model_size),
            mode="bilinear",
            align_corners=False
        )
        tensor_resized = tensor_resized.to(torch.float32).contiguous()
        blob = np.ascontiguousarray(tensor_resized.cpu().numpy())
        
        ratio_w = float(self.model_size) / float(w)
        ratio_h = float(self.model_size) / float(h)
        ratio = (ratio_w, ratio_h) if abs(ratio_w - ratio_h) > 1e-6 else float(ratio_w)
        scale = float(max(h, w)) / float(self.model_size)
        
        meta = {"mode": "stretch", "ratio": ratio, "pad_x": 0, "pad_y": 0}
        return blob, scale, meta

    def infer(self, blob: np.ndarray) -> np.ndarray:
        """
        Runs TensorRT inference on the preprocessed blob.
        
        Args:
            blob: Preprocessed image tensor
            
        Returns:
            Raw model output as numpy array
        """
        if not self.is_available:
            raise RuntimeError("TensorRTBackend is not ready.")
        
        # Get tensor shapes and types
        in_shape = tuple(self.ctx.get_tensor_shape(self.input_name))  # type: ignore[union-attr]
        out_shape = tuple(self.ctx.get_tensor_shape(self.output_name))  # type: ignore[union-attr]
        in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.input_name))  # type: ignore[union-attr]
        out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.output_name))  # type: ignore[union-attr]
        
        in_elems = int(np.prod(in_shape))
        out_elems = int(np.prod(out_shape))
        
        # Allocate host memory
        host_in = cuda.pagelocked_empty(in_elems, in_dtype)  # type: ignore[union-attr]
        host_out = cuda.pagelocked_empty(out_elems, out_dtype)  # type: ignore[union-attr]
        
        # Copy input data
        src = np.ascontiguousarray(blob).ravel().astype(in_dtype, copy=False)
        if src.size != in_elems:
            raise RuntimeError(f"Input size mismatch: engine expects {in_elems} elements, got {src.size}")
        
        np.copyto(np.frombuffer(host_in, dtype=in_dtype, count=in_elems), src)
        
        # Allocate device memory
        d_in = cuda.mem_alloc(host_in.nbytes)  # type: ignore[union-attr]
        d_out = cuda.mem_alloc(host_out.nbytes)  # type: ignore[union-attr]
        
        # Set tensor addresses
        self.ctx.set_tensor_address(self.input_name, int(d_in))  # type: ignore[union-attr]
        self.ctx.set_tensor_address(self.output_name, int(d_out))  # type: ignore[union-attr]
        
        # Run inference
        stream = cuda.Stream()  # type: ignore[union-attr]
        cuda.memcpy_htod_async(d_in, host_in, stream)  # type: ignore[union-attr]
        self.ctx.execute_async_v3(stream_handle=stream.handle)  # type: ignore[union-attr]
        cuda.memcpy_dtoh_async(host_out, d_out, stream)  # type: ignore[union-attr]
        stream.synchronize()
        
        # Process output
        trt_out = np.array(host_out).reshape(out_shape)
        if trt_out.dtype == np.float16:
            trt_out = trt_out.astype(np.float32)
        
        out_np = np.asarray(trt_out)
        if out_np.ndim == 3:
            if out_np.shape[0] == 1:
                out_np = np.squeeze(out_np, axis=0)
            else:
                raise RuntimeError("TRT output batch > 1 not supported")
        
        if out_np.ndim != 2:
            raise RuntimeError(f"Unexpected TRT output rank {out_np.ndim}; expected 2 after squeeze.")
        
        return out_np

    def decode_output(
        self,
        raw_output: np.ndarray,
        scale: float,
        meta: Dict[str, Any],
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ) -> List[Dict[str, Any]]:
        """
        Decodes TensorRT output into detection boxes with NMS.
        
        Args:
            raw_output: Raw output from TensorRT inference
            scale: Scale factor from preprocessing
            meta: Metadata from preprocessing
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            List of detections with bbox, confidence, class_id, class_name
        """
        # Transpose if needed: (84, N) -> (N, 84)
        if raw_output.shape[0] < raw_output.shape[1]:
            raw_output = raw_output.T
        
        # Extract box coordinates and class scores
        boxes = raw_output[:, :4]  # cx, cy, w, h
        class_scores = raw_output[:, 4:]  # class probabilities
        
        # Get best class for each detection
        class_ids = np.argmax(class_scores, axis=1)
        confidences = class_scores[np.arange(len(class_ids)), class_ids]
        
        # Filter by confidence threshold
        mask = confidences >= confidence_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        # Convert from (cx, cy, w, h) to (x1, y1, x2, y2)
        x1 = boxes[:, 0] - boxes[:, 2] / 2
        y1 = boxes[:, 1] - boxes[:, 3] / 2
        x2 = boxes[:, 0] + boxes[:, 2] / 2
        y2 = boxes[:, 1] + boxes[:, 3] / 2
        
        # Apply NMS
        indices = self._nms(x1, y1, x2, y2, confidences, iou_threshold)
        
        # Scale boxes back to original image size
        detections = []
        for idx in indices:
            bbox = [
                float(x1[idx] * scale),
                float(y1[idx] * scale),
                float(x2[idx] * scale),
                float(y2[idx] * scale)
            ]
            
            class_id = int(class_ids[idx])
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else str(class_id)
            
            detections.append({
                "bbox": bbox,
                "confidence": float(confidences[idx]),
                "class_id": class_id,
                "class_name": class_name
            })
        
        return detections

    def _nms(
        self,
        x1: np.ndarray,
        y1: np.ndarray,
        x2: np.ndarray,
        y2: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float
    ) -> List[int]:
        """Non-Maximum Suppression."""
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return keep
