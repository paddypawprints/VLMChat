"""Detection - lazy image crop with bbox and metadata."""

import threading
from typing import List, Tuple, Dict, Any, Optional
from enum import Enum


class CocoCategory(Enum):
    """Common COCO object categories."""
    PERSON = (0, "person")
    CAR = (2, "car")
    DOG = (16, "dog")
    CAT = (15, "cat")
    UNKNOWN = (-1, "unknown")
    
    def __init__(self, id: int, label: str):
        self.id = id
        self.label = label
    
    @classmethod
    def from_id(cls, id: int) -> Optional['CocoCategory']:
        """Get category by COCO ID."""
        for cat in cls:
            if cat.id == id:
                return cat
        return None


class ImageFormat(Enum):
    """Image data formats."""
    PIL = "pil"
    NUMPY = "numpy"
    TORCH = "torch"


class Detection:
    """
    Lazy image crop defined by bbox + source reference.
    
    Represents a region of interest without copying pixels until needed.
    Thread-safe caching for materialized crops.
    """
    
    def __init__(self,
                 bbox: Tuple[float, float, float, float],
                 confidence: float,
                 category: CocoCategory,
                 source_image: Any,
                 source_format: ImageFormat = ImageFormat.PIL):
        """
        Create detection.
        
        Args:
            bbox: (x1, y1, x2, y2) absolute coordinates
            confidence: Detection confidence (0.0 to 1.0)
            category: COCO category enum
            source_image: Source image reference (read-only)
            source_format: Format of source_image (default: PIL)
        """
        self.bbox = bbox
        self.confidence = confidence
        self.category = category
        self.source_image = source_image  # Read-only reference
        self.source_format = source_format
        self.children: List[Detection] = []
        self.metadata: Dict[str, Any] = {}  # Extensible metadata for task enrichment
        
        # Simple cache with lock for dict mutations
        self._cache: Dict[ImageFormat, Any] = {}
        self._lock = threading.Lock()
    
    def crop(self, format: ImageFormat = ImageFormat.NUMPY) -> Any:
        """
        Materialize crop on-demand (cached).
        
        Args:
            format: Desired image format
            
        Returns:
            Cropped image data
        """
        # Quick cache check (read-only, no lock)
        if format in self._cache:
            return self._cache[format]
        
        # Materialize crop
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        
        # Use source image directly (it's in source_format)
        source = self.source_image
        
        # Crop based on source format
        if self.source_format == ImageFormat.PIL:
            cropped = source.crop((x1, y1, x2, y2))
        elif self.source_format == ImageFormat.NUMPY:
            cropped = source[y1:y2, x1:x2].copy()
        else:
            raise ValueError(f"Unsupported source format: {self.source_format}")
        
        # Lock only for cache write
        with self._lock:
            if format not in self._cache:  # Double-check
                self._cache[format] = cropped
        
        return cropped
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize to dictionary (for JSON/MQTT).
        
        Returns:
            Dict with bbox, confidence, category, children, metadata
        """
        return {
            "bbox": list(self.bbox),
            "confidence": self.confidence,
            "category": self.category.label,
            "category_id": self.category.id,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (f"Detection(category={self.category.label}, "
                f"conf={self.confidence:.2f}, "
                f"bbox={self.bbox}, "
                f"children={len(self.children)})")
