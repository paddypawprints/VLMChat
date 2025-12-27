"""
Pipeline-native model implementations.

Simplified, self-contained model interfaces designed for pipeline integration.
No complex abstractions - just clean, direct interfaces.
"""

from .yolo_ultralytics import YoloUltralytics
from .yolo_tensorrt import YoloTensorRT

__all__ = [
    'YoloUltralytics',
    'YoloTensorRT',
]
