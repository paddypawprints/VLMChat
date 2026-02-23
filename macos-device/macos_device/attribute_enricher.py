"""Person attribute enrichment pipeline task (PA100K ONNX)."""

import logging
from pathlib import Path

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

from camera_framework import BaseTask
from camera_framework.detection import Detection, CocoCategory, ImageFormat
from .config import AttributesConfig
from .models.pa100k_onnx import PA100KOnnx

logger = logging.getLogger(__name__)


class AttributeEnricher(BaseTask):
    """
    Enriches person detections with PA100K attributes.

    Reads Detection objects from input buffer, runs attribute inference on each
    PERSON crop, and writes enriched detections downstream.

    Attributes are stored in detection.metadata:
        'attributes': Dict[str, {'value': bool, 'confidence': float}]

    Non-person detections pass through unmodified.
    """

    def __init__(self, config: AttributesConfig):
        """
        Initialize attribute enrichment task.

        Args:
            config: AttributesConfig (model path, confidence threshold, batch size)

        Raises:
            ValueError: If config is None
            ImportError: If onnxruntime or PIL not available
        """
        super().__init__(name="AttributeEnricher")

        if config is None:
            raise ValueError("AttributesConfig is required")

        if not PIL_AVAILABLE:
            raise ImportError("PIL not installed. Install with: pip install Pillow")

        self.config = config
        self.model_path = Path(config.model_path).expanduser()

        # Inference engine lazy-loaded on first process()
        self.engine: PA100KOnnx | None = None

        logger.info(f"AttributeEnricher initialised with model: {config.model_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_engine(self):
        if self.engine is not None:
            return
        if not self.model_path.exists():
            raise FileNotFoundError(f"PA100K model not found: {self.model_path}")
        logger.info(f"Loading PA100K model from: {self.model_path}")
        self.engine = PA100KOnnx(str(self.model_path))
        logger.info("PA100K model loaded")

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    def process(self):
        """Enrich person detections with attribute predictions."""
        try:
            self._load_engine()

            input_buffer = list(self.inputs.values())[0] if self.inputs else None
            if not input_buffer or not input_buffer.has_data():
                return

            message = input_buffer.get()
            if message is None:
                logger.debug("Got None from input buffer")
                return

            if not isinstance(message, dict):
                logger.warning(f"Invalid message format: {type(message)}")
                return

            detections = message.get("detections", [])

            if not detections:
                logger.debug("No detections in message")
                for buf in self.outputs.values():
                    buf.put(message)
                return

            logger.debug(f"Processing {len(detections)} detections")

            enriched_count = 0
            for detection in detections:
                if detection.category != CocoCategory.PERSON:
                    continue

                crop_pil = detection.crop(ImageFormat.PIL)
                attributes, probabilities = self.engine.predict(crop_pil)

                detection.metadata['category'] = detection.category.label
                detection.metadata['category_id'] = detection.category.id
                detection.metadata['attributes'] = {
                    attr: {'value': attributes[attr], 'confidence': float(probabilities[attr])}
                    for attr in attributes
                }
                enriched_count += 1

            logger.debug(f"Enriched {enriched_count}/{len(detections)} person detections")

            for buf in self.outputs.values():
                buf.put(message)

        except Exception as e:
            logger.error(f"Error in AttributeEnricher.process: {e}", exc_info=True)
            raise

    def __str__(self) -> str:
        return f"AttributeEnricher(model={self.model_path.name})"

    def __repr__(self) -> str:
        return self.__str__()
