"""
Object detection interface and detection result class.

This module defines the generic Detection class and the abstract interface
for object detection capabilities.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class Detection:
    """
    Represents a single object detection result.

    Encapsulates the results of object detection including bounding box coordinates,
    category classification, and confidence score. Coordinates are assumed to be
    already converted to image coordinates by the implementing camera class.
    """

    def __init__(self, box: Tuple[int, int, int, int], category: str, conf: float):
        """
        Create a Detection object with converted coordinates.

        Args:
            box: Bounding box coordinates (x1, y1, x2, y2) in image pixel coordinates
            category: Detected object category/class
            conf: Confidence score for the detection
        """
        self.box = box
        self.category = category
        self.conf = conf


class ObjectDetectionInterface(ABC):
    """
    Abstract interface for object detection capabilities.

    Defines the contract for cameras that support object detection,
    including detection parsing and continuous detection loops.
    """

    @abstractmethod
    def parse_detections(self, metadata: dict) -> Optional[List[Detection]]:
        """
        Parse neural network output into detected objects.

        Args:
            metadata: Frame metadata containing inference results

        Returns:
            List of detected objects or None if no outputs available
        """
        pass

    @abstractmethod
    def run_detection_loop(self):
        """
        Run continuous object detection loop.
        """
        pass

    @abstractmethod
    def get_labels(self) -> List[str]:
        """
        Get object detection labels.

        Returns:
            List of object class labels
        """
        pass