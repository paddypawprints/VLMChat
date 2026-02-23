"""YOLO object detection pipeline task (macOS / Ultralytics)."""

import logging
import numpy as np
from typing import List
from pathlib import Path

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

from camera_framework import BaseTask
from camera_framework.detection import Detection, CocoCategory, ImageFormat
from .config import YoloConfig

logger = logging.getLogger(__name__)


class YoloDetector(BaseTask):
    """
    YOLO object detection using Ultralytics (macOS optimised).

    Reads PIL Images from input buffer, produces Detection objects.
    """

    def __init__(self, config: YoloConfig):
        """
        Initialize YOLO detection task.

        Args:
            config: YoloConfig instance

        Raises:
            ImportError: If ultralytics is not installed
            ValueError: If config is None
        """
        super().__init__(name="YoloDetector")

        if not ULTRALYTICS_AVAILABLE:
            raise ImportError(
                "Ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )
        if config is None:
            raise ValueError("YoloConfig is required")

        self.model_path = Path(config.model_path).expanduser()
        self.confidence = config.confidence
        self.iou = config.iou
        self.device = config.device

        # Model loaded lazily on first process()
        self.model = None
        self.class_names: List[str] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_model(self):
        if self.model is not None:
            return

        if not self.model_path.exists():
            cwd_path = Path.cwd() / self.model_path.name
            if cwd_path.exists():
                self.model_path = cwd_path
            else:
                raise FileNotFoundError(f"YOLO model not found: {self.model_path}")

        logger.info(f"Loading YOLO model from: {self.model_path}")
        self.model = YOLO(str(self.model_path))

        # Warm up
        dummy = np.uint8(np.zeros((640, 640, 3)))
        self.model(dummy, device='cpu', verbose=False)

        self.class_names = [
            self.model.names[i] for i in range(len(self.model.names))
        ] if self.model.names else []
        logger.info(f"YOLO model loaded: {len(self.class_names)} classes")

    def _pil_to_numpy(self, pil_image):
        """Convert PIL Image to BGR numpy array."""
        rgb = np.array(pil_image)
        if len(rgb.shape) == 3 and rgb.shape[2] == 3:
            return rgb[:, :, ::-1].copy()
        return rgb

    def _class_id_to_category(self, class_id: int) -> CocoCategory:
        category_map = {
            0: CocoCategory.PERSON,
            2: CocoCategory.CAR,
            15: CocoCategory.CAT,
            16: CocoCategory.DOG,
        }
        return category_map.get(class_id, CocoCategory.UNKNOWN)

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def process(self):
        """Run YOLO detection on input frames and write detections downstream."""
        try:
            self._load_model()

            input_buffer = list(self.inputs.values())[0] if self.inputs else None
            if not input_buffer or not input_buffer.has_data():
                return

            message = input_buffer.get()
            if message is None:
                logger.warning("Got None from input buffer")
                return

            frames = message.get("frame", [])
            if not frames:
                logger.warning("No frames in message")
                return

            pil_image = frames[0]
            if pil_image is None:
                logger.warning("Frame is None")
                return

            logger.debug(
                f"Got image: type={type(pil_image)}, "
                f"size={pil_image.size if hasattr(pil_image, 'size') else 'N/A'}"
            )

            numpy_image = np.array(pil_image)
            logger.debug(f"Converted to numpy: shape={numpy_image.shape}, dtype={numpy_image.dtype}")

            if len(numpy_image.shape) != 3:
                logger.error(f"Invalid image shape: {numpy_image.shape}")
                return

            results = self.model(
                numpy_image,
                conf=self.confidence,
                iou=self.iou,
                device='cpu',
                verbose=False,
            )

        except Exception as e:
            logger.error(f"Error in YoloDetector.process: {e}", exc_info=True)
            raise

        detections: List[Detection] = []

        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().item())
                    cls  = int(box.cls[0].cpu().item())

                    if conf < self.confidence:
                        continue

                    detections.append(Detection(
                        bbox=(float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])),
                        confidence=conf,
                        category=self._class_id_to_category(cls),
                        source_image=pil_image,
                        source_format=ImageFormat.PIL,
                    ))

        # Synthetic full-frame detection ensures downstream buffers always have an image
        if not detections:
            w, h = pil_image.size
            detections.append(Detection(
                bbox=(0.0, 0.0, float(w), float(h)),
                confidence=0.0,
                category=CocoCategory.UNKNOWN,
                source_image=pil_image,
                source_format=ImageFormat.PIL,
            ))
            logger.debug("No detections — created synthetic full-frame detection")
        else:
            logger.debug(f"YOLO detected {len(detections)} objects")

        out_message = {"detections": detections}
        for buf in self.outputs.values():
            buf.put(out_message)

    def __str__(self) -> str:
        return f"YoloDetector(model={self.model_path.name}, conf={self.confidence})"

    def __repr__(self) -> str:
        return self.__str__()
