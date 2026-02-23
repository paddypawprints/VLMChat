"""macOS device application for camera pipeline testing."""

from .camera import Camera
from .yolo_detector import YoloDetector
from .attribute_enricher import AttributeEnricher
from .smolvlm_verifier import SmolVLMVerifier

__version__ = "0.1.0"

__all__ = ['Camera', 'YoloDetector', 'AttributeEnricher', 'SmolVLMVerifier', '__version__']
