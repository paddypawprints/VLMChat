"""
ONNX Runtime backend for PA100K person attribute detection.

Predicts 26 pedestrian attributes (gender, age, clothing, accessories, etc.)
from a cropped person image.
"""
from __future__ import annotations

import logging
from typing import Dict, Tuple

import numpy as np

try:
    import onnxruntime as ort
    ONNXRUNTIME_AVAILABLE = True
except ImportError:
    ONNXRUNTIME_AVAILABLE = False
    ort = None

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

logger = logging.getLogger(__name__)

# PA100K attribute labels — order must match training
PA100K_ATTRIBUTES = [
    "Female", "AgeOver60", "Age18-60", "AgeLess18", "Front", "Side", "Back",
    "Hat", "Glasses", "HandBag", "ShoulderBag", "Backpack", "HoldObjectsInFront",
    "ShortSleeve", "LongSleeve", "UpperStride", "UpperLogo", "UpperPlaid", "UpperSplice",
    "LowerStripe", "LowerPattern", "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots",
]


class PA100KOnnx:
    """
    ONNX Runtime backend for PA100K person attribute detection.

    Accepts a cropped person PIL Image, returns per-attribute boolean signals
    and raw sigmoid probabilities.
    """

    def __init__(self, model_path: str):
        """
        Initialize ONNX inference session.

        Args:
            model_path: Path to PA100K ONNX model file

        Raises:
            ImportError: If onnxruntime is not installed
        """
        if not ONNXRUNTIME_AVAILABLE:
            raise ImportError("onnxruntime not installed. Install with: pip install onnxruntime")

        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.attributes = PA100K_ATTRIBUTES

        logger.info(f"PA100KOnnx loaded: {len(self.attributes)} attributes from {model_path}")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _preprocess(self, pil_img: "Image.Image") -> np.ndarray:
        """Letterbox-pad to 224×224 and normalise with ImageNet stats."""
        target = 224
        w, h = pil_img.size
        scale = min(target / w, target / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Pad with ImageNet mean colour
        padded = Image.new('RGB', (target, target), color=(123, 117, 104))
        padded.paste(resized, ((target - new_w) // 2, (target - new_h) // 2))

        img = np.array(padded).transpose(2, 0, 1).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        return (img - mean / std)[np.newaxis].astype(np.float32)

    def _apply_logic(self, logits: np.ndarray) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """
        Apply post-processing logic to raw logits.

        Returns:
            (signals, probs) where signals are final bool decisions and
            probs are raw sigmoid probabilities.
        """
        probs = 1 / (1 + np.exp(-logits))
        raw = dict(zip(self.attributes, probs))
        signals: Dict[str, bool] = {}

        # Mutually exclusive age group
        age_group = ["AgeLess18", "Age18-60", "AgeOver60"]
        best_age = max(age_group, key=lambda x: raw[x])
        for a in age_group:
            signals[a] = (a == best_age)

        # Mutually exclusive orientation
        view_group = ["Front", "Side", "Back"]
        best_view = max(view_group, key=lambda x: raw[x])
        for v in view_group:
            signals[v] = (v == best_view)

        # Binary attributes at 0.5 threshold
        binary_attrs = [
            "Female", "Hat", "Glasses", "HandBag", "ShoulderBag", "Backpack",
            "HoldObjectsInFront", "ShortSleeve", "LongSleeve", "UpperStride",
            "UpperLogo", "UpperPlaid", "UpperSplice", "LowerStripe", "LowerPattern",
            "LongCoat", "Trousers", "Shorts", "Skirt&Dress", "boots",
        ]
        for attr in binary_attrs:
            signals[attr] = raw[attr] > 0.5

        return signals, raw

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, pil_img: "Image.Image") -> Tuple[Dict[str, bool], Dict[str, float]]:
        """
        Run inference on a cropped person PIL Image.

        Args:
            pil_img: Cropped person image

        Returns:
            Tuple of (attribute_signals, raw_probabilities)
        """
        input_tensor = self._preprocess(pil_img)
        logits = self.session.run(None, {self.input_name: input_tensor})[0][0]
        return self._apply_logic(logits)

    def __repr__(self) -> str:
        return f"PA100KOnnx(attributes={len(self.attributes)})"
