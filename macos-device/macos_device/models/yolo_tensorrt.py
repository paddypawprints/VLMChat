"""
TensorRT backend for GPU-accelerated YOLO inference (Jetson / CUDA devices).

Self-contained: only depends on camera_framework, numpy, and cv2.
TensorRT + pycuda are imported lazily so the module loads safely on CPU-only machines.
"""
from __future__ import annotations

import logging
import numpy as np
from typing import List, Optional
from pathlib import Path

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit  # noqa: F401 – initialises CUDA context
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    trt = None
    cuda = None

from camera_framework import Detection, CocoCategory, ImageFormat

logger = logging.getLogger(__name__)

# Standard COCO-80 class names
_COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush',
]


class YoloTensorRT:
    """
    TensorRT-optimised YOLO backend for GPU inference (Jetson / CUDA).

    Accepts raw numpy BGR images directly — no vlmchat ImageContainer needed.
    Returns camera_framework Detection objects ready for pipeline use.
    """

    def __init__(
        self,
        engine_path: str,
        class_names: Optional[List[str]] = None,
    ):
        """
        Initialize TensorRT YOLO model.

        Args:
            engine_path: Path to TensorRT engine file (.engine or .trt)
            class_names: Class names (defaults to COCO-80)

        Raises:
            ImportError: If tensorrt / pycuda are not installed
            FileNotFoundError: If engine_path does not exist
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError(
                "TensorRT not available. "
                "Install onnxruntime-gpu or tensorrt+pycuda for Jetson."
            )

        self.engine_path = Path(engine_path).expanduser()
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {self.engine_path}")

        logger.info(f"Loading TensorRT engine: {self.engine_path}")
        self.engine, self.context = self._load_engine()
        self._setup_bindings()

        self.class_names = class_names or _COCO_CLASSES
        logger.info(f"TensorRT YOLO ready ({len(self.class_names)} classes)")

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __del__(self):
        try:
            if hasattr(self, 'inputs'):
                for inp in self.inputs:
                    if inp.get('device'):
                        inp['device'].free()
            if hasattr(self, 'outputs'):
                for out in self.outputs:
                    if out.get('device'):
                        out['device'].free()
            if hasattr(self, 'context') and self.context:
                del self.context
            if hasattr(self, 'engine') and self.engine:
                del self.engine
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_engine(self):
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        with open(self.engine_path, 'rb') as f:
            engine_data = f.read()
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(engine_data)
        context = engine.create_execution_context()
        return engine, context

    def _setup_bindings(self):
        self.bindings: List[int] = []
        self.inputs: List[dict] = []
        self.outputs: List[dict] = []

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))

            size = trt.volume(shape)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            self.bindings.append(int(device_mem))
            entry = {'name': name, 'shape': shape, 'dtype': dtype,
                     'host': host_mem, 'device': device_mem}

            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append(entry)
            else:
                self.outputs.append(entry)

        logger.info(
            f"TensorRT bindings: input={self.inputs[0]['shape']}, "
            f"output={self.outputs[0]['shape']}"
        )

    def _preprocess(self, image: np.ndarray) -> tuple:
        """Letterbox resize + normalise to float32 CHW."""
        import cv2
        _, _, height, width = self.inputs[0]['shape']  # BCHW
        orig_h, orig_w = image.shape[:2]

        r = min(height / orig_h, width / orig_w)
        new_h, new_w = int(orig_h * r), int(orig_w * r)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        dh, dw = height - new_h, width - new_w
        top, left = dh // 2, dw // 2
        padded = cv2.copyMakeBorder(
            resized, top, dh - top, left, dw - left,
            cv2.BORDER_CONSTANT, value=(114, 114, 114),
        )

        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        batched = np.expand_dims(rgb.transpose(2, 0, 1).astype(np.float32) / 255.0, 0)
        meta = {'ratio': r, 'pad': (top, left), 'orig_shape': (orig_h, orig_w)}
        return batched, r, meta

    def _postprocess(self, output: np.ndarray, meta: dict,
                     confidence: float, iou: float) -> List[tuple]:
        """Convert raw YOLO output to (bbox, score, class_id) tuples."""
        import cv2

        # Handle [B, 84, N] or [B, N, 84]
        if output.shape[1] == 84:
            predictions = output[0].T
        elif output.shape[2] == 84:
            predictions = output[0]
        else:
            logger.error(f"Unexpected output shape: {output.shape}")
            return []

        boxes_cxcywh = predictions[:, :4]
        class_scores = predictions[:, 4:]
        scores = np.max(class_scores, axis=1)
        class_ids = np.argmax(class_scores, axis=1)

        mask = scores >= confidence
        if not mask.any():
            return []

        boxes_cxcywh, scores, class_ids = boxes_cxcywh[mask], scores[mask], class_ids[mask]

        # cx,cy,w,h → x1,y1,x2,y2
        x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
        y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
        x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
        y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2

        top, left = meta['pad']
        r = meta['ratio']
        orig_h, orig_w = meta['orig_shape']

        x1 = np.clip((x1 - left) / r, 0, orig_w)
        y1 = np.clip((y1 - top) / r, 0, orig_h)
        x2 = np.clip((x2 - left) / r, 0, orig_w)
        y2 = np.clip((y2 - top) / r, 0, orig_h)

        indices = cv2.dnn.NMSBoxes(
            np.stack([x1, y1, x2, y2], axis=1).tolist(),
            scores.tolist(), confidence, iou,
        )
        if len(indices) == 0:
            return []

        return [
            ((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])),
             float(scores[i]),
             int(class_ids[i]))
            for i in indices.flatten()
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self.engine is not None and self.context is not None

    def detect(
        self,
        image: np.ndarray,
        confidence: float = 0.25,
        iou: float = 0.45,
        source_image=None,
    ) -> List[Detection]:
        """
        Run YOLO detection on a BGR numpy image.

        Args:
            image: Input image as numpy array (HWC, BGR, uint8)
            confidence: Confidence threshold
            iou: NMS IoU threshold
            source_image: Optional source object stored on Detection (e.g. PIL Image or frame ID)

        Returns:
            List of camera_framework Detection objects
        """
        if not self.is_available:
            raise RuntimeError("TensorRT engine not loaded")

        preprocessed, _, meta = self._preprocess(image)

        np.copyto(self.inputs[0]['host'], preprocessed.ravel())
        cuda.memcpy_htod(self.inputs[0]['device'], self.inputs[0]['host'])
        self.context.execute_v2(self.bindings)
        cuda.memcpy_dtoh(self.outputs[0]['host'], self.outputs[0]['device'])
        output = self.outputs[0]['host'].reshape(self.outputs[0]['shape'])

        detections = []
        for bbox, conf, class_id in self._postprocess(output, meta, confidence, iou):
            category = CocoCategory.from_id(class_id)
            if category is None:
                logger.warning(f"Unknown COCO class ID {class_id}, skipping")
                continue
            detections.append(Detection(
                bbox=bbox,
                confidence=conf,
                category=category,
                source_image=source_image,
            ))

        logger.debug(f"TensorRT YOLO: {len(detections)} detections")
        return detections

    def __repr__(self) -> str:
        return f"YoloTensorRT(engine={self.engine_path.name}, classes={len(self.class_names)})"
