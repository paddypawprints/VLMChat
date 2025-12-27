"""
Detection class - lazy ImageContainer defined by bbox + source reference.

Detections extend ImageContainer to represent cropped regions that are
materialized on-demand. Supports hierarchical structure for clustering.
"""
from typing import Optional, List, Tuple, Dict, Any, TYPE_CHECKING
from collections import defaultdict
import numpy as np
import itertools
import logging

from .cache.image import ImageContainer
from .image.formats import ImageFormat
from .filters import ImageFilter
from .categories import CocoCategory

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Global counter for detection IDs
_GLOBAL_DETECTION_ID_COUNTER = itertools.count(1)


class Detection(ImageContainer):
    """
    Detection is a lazy ImageContainer defined by bbox + source reference.
    
    Represents a cropped region of a source image, materialized on-demand
    when get() is called. Can have children forming a hierarchy for clustering.
    
    All bbox coordinates are absolute (relative to root source_image).
    """
    
    def __init__(self,
                 bbox: Tuple[float, float, float, float],
                 confidence: float,
                 category: CocoCategory,
                 source_image: ImageContainer,
                 cache_key: Optional[str] = None):
        """
        Create detection as virtual ImageContainer.
        
        Args:
            bbox: (x1, y1, x2, y2) in absolute coordinates relative to source_image
            confidence: Detection confidence score (0.0 to 1.0)
            category: COCO category enum
            source_image: ImageContainer to crop from (the root source)
            cache_key: Optional cache key (auto-generated if None)
        """
        # Generate ID and cache key
        det_id = next(_GLOBAL_DETECTION_ID_COUNTER)
        key = cache_key or f"det_{det_id}"
        
        # Initialize as ImageContainer (no source_data yet - lazy)
        super().__init__(key)
        
        # Detection-specific fields
        self.bbox = bbox
        self.confidence = confidence
        self.category = category
        self.source_image = source_image
        self.id = det_id
        
        # Hierarchy support
        self.children: List['Detection'] = []
        
        # Optional filter for privacy/rendering
        self.image_filter: Optional[ImageFilter] = None
        
        # Optional CLIP labeling results
        self.matched_prompts: Optional[List[str]] = None
        self.match_probabilities: Optional[List[float]] = None
        
        # Optional tracking ID
        self.track_id: Optional[int] = None
        
        # Inherit source format from parent image
        self.source_format = source_image.source_format
        
        # Extract dimensions from bbox
        self._width = int(bbox[2] - bbox[0])
        self._height = int(bbox[3] - bbox[1])
    
    @property
    def class_name(self) -> str:
        """Backward compatibility: return category label."""
        return self.category.label
    
    @property
    def class_id(self) -> int:
        """Backward compatibility: return category ID."""
        return self.category.id
    
    def add_child(self, detection: 'Detection') -> None:
        """
        Add child detection to hierarchy.
        
        Child inherits source_image if not already set.
        All bbox coords remain absolute (relative to root source).
        
        Args:
            detection: Child Detection to add
        """
        if detection.source_image is None:
            detection.source_image = self.source_image
        self.children.append(detection)
    
    def get(self, format: Optional[ImageFormat] = None) -> Any:
        """
        Get cropped image data (lazy materialization).
        
        Overrides ImageContainer.get() to crop from source_image.
        Caches the result for subsequent calls.
        
        Args:
            format: Desired image format (None = source format)
        
        Returns:
            Cropped image data in requested format
        """
        fmt = format or self.source_format
        
        # Optimistic read - check cache first (no lock)
        if fmt in self._formats:
            return self._formats[fmt]
        
        # Lazy crop from source
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        
        # Clamp to source image bounds
        source_img = self.source_image.get(self.source_format)
        if self.source_format == ImageFormat.PIL:
            max_w, max_h = source_img.size
        elif self.source_format == ImageFormat.NUMPY:
            max_h, max_w = source_img.shape[:2]
        else:
            max_h, max_w = None, None
        
        if max_w and max_h:
            x1 = max(0, min(x1, max_w))
            x2 = max(0, min(x2, max_w))
            y1 = max(0, min(y1, max_h))
            y2 = max(0, min(y2, max_h))
        
        # Crop based on source format
        if self.source_format == ImageFormat.PIL:
            cropped = source_img.crop((x1, y1, x2, y2))
        elif self.source_format == ImageFormat.NUMPY:
            cropped = source_img[y1:y2, x1:x2].copy()
        elif self.source_format in (ImageFormat.TORCH_CPU, ImageFormat.TORCH_GPU):
            cropped = source_img[:, y1:y2, x1:x2].clone()
        else:
            raise ValueError(f"Unsupported source format: {self.source_format}")
        
        # Apply image filter if present (for privacy/rendering)
        if self.image_filter and self.source_format == ImageFormat.NUMPY:
            # Filter operates in-place on numpy array
            crop_bbox = (0, 0, x2 - x1, y2 - y1)
            self.image_filter.apply(cropped, crop_bbox)
        
        # Cache the cropped result in source format
        with self._rwlock.gen_wlock():
            # Double-check another thread didn't just do this
            if self.source_format in self._formats:
                cropped = self._formats[self.source_format]
            else:
                self._formats[self.source_format] = cropped
        
        # Convert to requested format if different
        if fmt != self.source_format:
            logger.debug(f"Detection converting {self.source_format} -> {fmt}")
            cropped = self._convert_format(fmt)
            logger.debug(f"Detection conversion result: {type(cropped)}")
        
        return cropped
    
    def materialize(self, format: Optional[ImageFormat] = None) -> Any:
        """
        Materialize detection as cropped image (explicit interface).
        
        Alias for get() - provides explicit materialization interface
        that's consistent between ImageContainer and Detection.
        
        Args:
            format: Desired image format
        
        Returns:
            Cropped image data in requested format
        """
        return self.get(format)
    
    def get_base_image(self) -> ImageContainer:
        """
        Get the root source ImageContainer.
        
        Returns the full image that this detection (and its children) reference.
        Useful for visualization where we need to draw bboxes on the full image.
        
        Returns:
            Root source ImageContainer
        """
        return self.source_image
    
    @staticmethod
    def get_common_base_image(detections: List['Detection']) -> ImageContainer:
        """
        Get the common base image from a list of detections.
        
        Verifies all detections reference the same source.
        
        Args:
            detections: List of Detection objects
        
        Returns:
            Common source ImageContainer
        
        Raises:
            ValueError: If detections reference different sources or list is empty
        """
        if not detections:
            raise ValueError("Empty detection list")
        
        base = detections[0].source_image
        
        # Verify all detections share same source
        for det in detections[1:]:
            if det.source_image is not base:
                raise ValueError("Detections reference different source images")
        
        return base
    
    @staticmethod
    def group_by_source(detections: List['Detection']) -> Dict[ImageContainer, List['Detection']]:
        """
        Group detections by their source image.
        
        Useful for multi-camera scenarios where detections come from
        different source images.
        
        Args:
            detections: List of Detection objects
        
        Returns:
            Dictionary mapping source_image -> list of detections
        """
        grouped = defaultdict(list)
        
        for det in detections:
            if det.source_image is None:
                raise ValueError(f"Detection {det.id} has no source_image")
            grouped[det.source_image].append(det)
        
        return dict(grouped)
    
    def draw_on_image(self, 
                     image: np.ndarray,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2,
                     recursive: bool = True,
                     depth: int = 0,
                     apply_filters: bool = True) -> np.ndarray:
        """
        Draw this detection (and optionally children) on an image.
        
        Args:
            image: Numpy array image to draw on (RGB, modified in-place)
            color: RGB color tuple for bbox
            thickness: Line thickness for bbox
            recursive: Whether to draw children recursively
            depth: Current recursion depth (for coloring)
            apply_filters: Whether to apply image_filter to detection region
        
        Returns:
            Image with detection drawn (modified in-place and returned)
        """
        import cv2
        
        x1, y1, x2, y2 = [int(v) for v in self.bbox]
        
        # Apply filter first (if present and enabled)
        if apply_filters and self.image_filter:
            self.image_filter.apply(image, (x1, y1, x2, y2))
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        
        # Build label
        label = f"[{self.id}] {self.class_name} {self.confidence:.2f}"
        if self.image_filter:
            label += " [F]"  # Indicate filter applied
        if depth > 0:
            label += f" [d:{depth}]"
        if self.children:
            label += f" [{len(self.children)}]"
        
        # Draw text with background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
        
        # Text background rectangle
        cv2.rectangle(image, (x1, y1 - text_h - baseline - 2), 
                     (x1 + text_w, y1), color, -1)
        
        # Text
        cv2.putText(image, label, (x1, y1 - baseline - 2), 
                   font, font_scale, (255, 255, 255), text_thickness)
        
        # Draw children recursively
        if recursive and self.children:
            # Vary color by depth
            child_color = (
                (color[0] + 100) % 256,
                (color[1] + 100) % 256,
                (color[2] + 100) % 256
            )
            child_thickness = max(1, thickness - 1)
            
            for child in self.children:
                image = child.draw_on_image(
                    image, child_color, child_thickness,
                    recursive, depth + 1, apply_filters
                )
        
        return image
    
    @staticmethod
    def visualize_tiled(detections: List['Detection'],
                       format: ImageFormat = ImageFormat.NUMPY,
                       show_hierarchy: bool = True,
                       apply_filters: bool = True,
                       tile_layout: Optional[Tuple[int, int]] = None) -> Any:
        """
        Create tiled visualization with crops drawn on their source images.
        
        Groups detections by source image, draws each group on its source,
        then tiles the annotated sources together in a grid.
        
        Args:
            detections: List of Detection objects to visualize
            format: Output image format
            show_hierarchy: Whether to draw children recursively
            apply_filters: Whether to apply image_filter to filtered detections
            tile_layout: Optional (rows, cols) for tiling. If None, auto-calculate.
        
        Returns:
            Tiled image with all sources and their detections drawn
        """
        import cv2
        
        if not detections:
            raise ValueError("No detections to visualize")
        
        # Group by source image
        grouped = Detection.group_by_source(detections)
        sources = list(grouped.keys())
        num_sources = len(sources)
        
        if num_sources == 0:
            raise ValueError("No source images found")
        
        # Calculate tile layout
        if tile_layout:
            rows, cols = tile_layout
        else:
            # Auto-calculate: roughly square grid
            cols = int(np.ceil(np.sqrt(num_sources)))
            rows = int(np.ceil(num_sources / cols))
        
        # Colors for different detections
        colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
        
        # Render each source with its detections
        tiles = []
        for source in sources:
            # Get source image as numpy
            source_img = source.materialize(ImageFormat.NUMPY).copy()
            
            # Draw all detections for this source
            dets = grouped[source]
            for i, det in enumerate(dets):
                color = colors[i % len(colors)]
                source_img = det.draw_on_image(
                    source_img, color,
                    thickness=2,
                    recursive=show_hierarchy,
                    apply_filters=apply_filters
                )
            
            tiles.append(source_img)
        
        # Get max dimensions for consistent tile sizing
        max_h = max(tile.shape[0] for tile in tiles)
        max_w = max(tile.shape[1] for tile in tiles)
        
        # Pad tiles to same size
        padded_tiles = []
        for tile in tiles:
            h, w = tile.shape[:2]
            if h < max_h or w < max_w:
                # Pad with black
                padded = np.zeros((max_h, max_w, 3), dtype=tile.dtype)
                padded[:h, :w] = tile
                padded_tiles.append(padded)
            else:
                padded_tiles.append(tile)
        
        # Create tiled grid
        tile_rows = []
        for r in range(rows):
            row_tiles = []
            for c in range(cols):
                idx = r * cols + c
                if idx < len(padded_tiles):
                    row_tiles.append(padded_tiles[idx])
                else:
                    # Fill empty slots with black
                    row_tiles.append(np.zeros((max_h, max_w, 3), dtype=padded_tiles[0].dtype))
            tile_rows.append(np.hstack(row_tiles))
        
        # Stack rows vertically
        tiled_image = np.vstack(tile_rows)
        
        # Convert to requested format
        if format != ImageFormat.NUMPY:
            from .image.converter import ImageFormatConverter
            converter = ImageFormatConverter()
            tiled_image = converter.convert(tiled_image, ImageFormat.NUMPY, format)
        
        return tiled_image
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"Detection(id={self.id}, class={self.class_name}, "
                f"conf={self.confidence:.2f}, bbox={self.bbox}, "
                f"children={len(self.children)})")
