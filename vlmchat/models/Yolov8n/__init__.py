"""
YOLOv8n model module.

Provides object detection capabilities using YOLOv8 nano model with TensorRT backend.
"""

from .yolo_model import YoloModel
from .runtime_base import YoloRuntimeBase
from .tensorrt_backend import TensorRTBackend
from .ultralytics_backend import UltralyticsBackend

__all__ = ["YoloModel", "YoloRuntimeBase", "TensorRTBackend", "UltralyticsBackend"]
