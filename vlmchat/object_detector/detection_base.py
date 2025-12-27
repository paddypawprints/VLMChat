"""
Object detection interface and detection result class.

This module defines the generic Detection class and the abstract interface
for object detection capabilities, which can be chained in a pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Callable, Any # Added Callable
from PIL import Image
import math
import itertools # Added

# --- Global ID Counter ---
# Use itertools.count(1) to create a thread-safe, static counter
_GLOBAL_DETECTION_ID_COUNTER = itertools.count(1)

# --- Global utility for compare() ---
def _are_lists_equivalent(list1: List['Detection'], list2: List['Detection']) -> bool:
    """
    Checks if two lists of Detections are equivalent using .compare().
    Order does not matter.
    """
    if len(list1) != len(list2):
        return False
    list2_matches = [False] * len(list2)
    for obj1 in list1:
        match_found = False
        for i, obj2 in enumerate(list2):
            if not list2_matches[i] and obj1.compare(obj2):
                list2_matches[i] = True
                match_found = True
                break
        if not match_found:
            return False
    return True

class Detection:
    """
    Represents a single object detection result.

    Encapsulates the results of object detection including bounding box coordinates,
    category classification, and confidence score. Coordinates are assumed to be
    already converted to image coordinates by the implementing camera class.
    
    A unique `id` is automatically assigned upon creation.
    """

    def __init__(self, 
                 box: Tuple[int, int, int, int], 
                 object_category: str, 
                 detection_category: str = "Object", 
                 conf: float = 0.0):
        """
        Create a Detection object with converted coordinates.

        Args:
            box: Bounding box coordinates (x1, y1, x2, y2) in image pixel coordinates
            object_category: Detected object category/class
            detection_category: The type of detection (e.g., "Object", "Face")
            conf: Confidence score for the detection
        """
        self.id: int = next(_GLOBAL_DETECTION_ID_COUNTER) # Auto-assign ID
        self.box = box
        self.object_category = object_category
        self.conf = conf
        self.children: List['Detection'] = []
        
        # Optional attributes for CLIP/FashionCLIP labeling
        self.matched_prompts: Optional[List[str]] = None
        self.match_probabilities: Optional[List[float]] = None

    @classmethod
    def deserialize(cls, data: dict) -> 'Detection':
        """
        Populate the Detection object from a dictionary.
        
        Note: This does *not* preserve the original ID, it assigns a new one.
        This is intended for loading test data, not for full serialization.

        Args:
            data: Dictionary containing detection data
        """
        box = tuple(data['box'])
        object_category = data['object_category']
        conf = data['conf']
        # Note: We skip 'id' from the data, a new one will be auto-assigned.
        children = [Detection.deserialize(child) for child in data.get('children', [])]

        # Call __init__ which will assign a new ID.
        detection = cls(box, object_category, conf=conf)
        detection.children = children
        
        # Load optional matched prompts and probabilities if present
        if 'matched_prompts' in data:
            detection.matched_prompts = data['matched_prompts']
        if 'match_probabilities' in data:
            detection.match_probabilities = data['match_probabilities']
        
        return detection

    def add_child(self, detection: 'Detection' ) -> None:
        self.children.append(detection)
    
    def to_dict(self) -> dict:
        """
        Serialize Detection to dictionary for pickling/serialization.
        
        Returns:
            Dictionary with all detection attributes including children
        """
        return {
            'id': self.id,
            'box': self.box,
            'object_category': self.object_category,
            'conf': self.conf,
            'matched_prompts': self.matched_prompts,
            'match_probabilities': self.match_probabilities,
            'children': [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Detection':
        """
        Reconstruct Detection from dictionary.
        
        Note: Restores original ID (unlike deserialize which assigns new ID).
        
        Args:
            data: Dictionary from to_dict()
        
        Returns:
            Detection instance with original ID preserved
        """
        det = cls(
            box=tuple(data['box']),
            object_category=data['object_category'],
            conf=data['conf']
        )
        # Restore original ID (important for deduplication)
        det.id = data['id']
        det.matched_prompts = data.get('matched_prompts')
        det.match_probabilities = data.get('match_probabilities')
        det.children = [cls.from_dict(c) for c in data.get('children', [])]
        return det

    def __repr__(self) -> str:
        """
        Returns an unambiguous, official string representation for debugging.
        """
        return f"Detection(id={self.id}, box={self.box}, cat='{self.object_category}', conf={self.conf:.2f})"
    
    def __str__(self) -> str:
        """
        Returns an informal, human-readable string representation.
        """
        child_count = len(self.children)
        base_str = f"[ID {self.id}] '{self.object_category}' (conf: {self.conf:.2f}) at {self.box}"
        
        if child_count > 0:
            base_str += f" [{child_count} children]"
        
        # Add matched prompts if available
        if self.matched_prompts and self.match_probabilities:
            matches_str = ", ".join([f"{p} ({prob:.3f})" for p, prob in 
                                     zip(self.matched_prompts, self.match_probabilities)])
            base_str += f" | Matches: {matches_str}"
        
        return base_str


    def compare(self, other: Any) -> bool: # Renamed from is_equivalent
        """
        Compares this Detection object to another.
        
        Returns True if all scalar attributes (box, category, conf) are
        equal and their children are equivalent (regardless of order).
        
        The 'id' field is *ignored* for comparison.

        Args:
            other: The other object to compare against.

        Returns:
            bool: True if the objects are equivalent, False otherwise.
        """
        if not isinstance(other, Detection):
            return False

        # 1. Compare scalar attributes
        # Use math.isclose for robust float comparison
        scalars_match = (
            self.box == other.box and
            self.object_category == other.object_category and
            math.isclose(self.conf, other.conf, rel_tol=1e-5)
        )
        
        if not scalars_match:
            return False

        # 2. Compare children (order-agnostic)
        # Uses the helper function defined at the top of the file
        return _are_lists_equivalent(self.children, other.children)

# --- Utility Functions for Merging ---

def _calculate_iou(box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
    """
    Calculates the Intersection over Union (IoU) for two bounding boxes.

    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)

    Returns:
        The IoU score as a float between 0.0 and 1.0.
    """
    # Determine the coordinates of the intersection rectangle
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # Calculate intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height

    # Calculate union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area

def merge_detections(
    base_list: List[Detection], 
    new_list: List[Detection], 
    should_merge: Callable[[Detection, Detection], bool]
) -> List[Detection]:
    """
    Merges a new list of detections into a base list using a filter function.

    This function implements a form of Non-Maximal Suppression (NMS)
    where the "duplicate" logic is defined by the 'should_merge' function.
    If a duplicate is found, the one with the higher confidence score is kept.

    Args:
        base_list: The primary list of detections.
        new_list: The new detections to merge in.
        should_merge: A function(new_det, base_det) -> bool.
                      Returns True if the two detections are considered
                      a match/duplicate, False otherwise.

    Returns:
        A new, merged list of detections.
    """
    final_list = list(base_list)
    detections_to_add: List[Detection] = []

    for new_det in new_list:
        match_found = False
        for i, base_det in enumerate(final_list):
            if should_merge(new_det, base_det):
                match_found = True
                # Match found, keep the one with higher confidence
                if new_det.conf > base_det.conf:
                    final_list[i] = new_det  # Replace
                # If base_det.conf is higher, we do nothing (drop new_det)
                break  # Stop checking for this new_det
        
        if not match_found:
            detections_to_add.append(new_det)

    final_list.extend(detections_to_add)
    return final_list

# --- End Utility Functions ---


class ObjectDetector(ABC):
    """
    Abstract interface for object detection capabilities.

    Defines the contract for detectors, which can be chained together
    to form a processing pipeline.
    """
    
    def __init__(self, source: Optional['ObjectDetector'] = None):
        """
        Initializes the detector, optionally chaining it to a source detector.
        
        Args:
            source: An optional preceding detector in the pipeline.
        """
        self._source = source

    def start(self, audit: bool = False) -> None: # Added audit flag
        """
        Starts and initializes the detector and its source (if any).
        Implementors should override this, call super().start(), 
        and then add their own initialization logic.
        """
        if self._source:
            self._source.start(audit) # Pass audit flag down

    def stop(self) -> None:
        """
        Stops and cleans up the detector and its source (if any).
        Implementors should override this, call super().stop(),
        and then add their own cleanup logic.
        """
        if self._source:
            self._source.stop()

    @abstractmethod
    def readiness(self) -> bool:
        """
        Returns True if the detector is initialized and ready.
        Implementors must check their own readiness and that of their source.
        
        Example:
            source_ready = super().readiness()
            return self._my_model_loaded and source_ready
        """
        if self._source:
            return self._source.readiness()
        # If no source, child must override and return its own status
        return False

    @abstractmethod
    def _detect_internal(self, image: Image, detections: List[Detection]) -> List[Detection]:
        """
        Internal detection logic for this specific detector.
        
        Implementors should perform their detection/filtering logic on the 
        provided list of detections, modifying it or adding to it as needed.

        Args:
            image: The input PIL Image.
            detections: The list of detections from the source (if any). 
                        This list can be modified in-place or a new list returned.

        Returns:
            The final list of Detection objects for this stage.
        """
        pass

    def detect(self, image: Image, detections: Optional[List[Detection]] = None) -> List[Detection]:
        """
        Performs object detection on a single image.
        If a source detector is chained, it will be called first.

        Args:
            image: The input PIL Image.
            detections: An optional existing list to append new detections to.
                        This is primarily for the *start* of the chain.

        Returns:
            The final list of Detection objects from this detector's stage.
        """
        # 1. Get initial detections from the source, if it exists.
        if self._source:
            # Pass the image and any initial detections down the chain
            detection_list = self._source.detect(image, detections)
        elif detections is not None:
            # This is the start of the chain, use the provided list
            detection_list = detections
        else:
            # This is the start of the chain, create a new list
            detection_list = []
        
        # 2. Perform this detector's specific logic on the list
        return self._detect_internal(image, detection_list)

    def get_labels(self) -> List[str]:
        """
        Get object detection labels.
        By default, returns labels from the source.
        Implementors can override to add/modify labels.
        
        Example (for a producer like YOLO):
            source_labels = super().get_labels()
            return source_labels + [l for l in self._my_labels if l not in source_labels]
        """
        if self._source:
            return self._source.get_labels()
        return []