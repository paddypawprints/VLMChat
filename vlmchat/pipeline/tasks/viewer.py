"""
Viewer task for visualizing detections.

Renders images with detection bounding boxes overlaid, tiles multiple images
horizontally, and saves to file.
"""

import logging
import os
import time
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np

from ..core.task_base import BaseTask, Context, ContextDataType, register_task
from ..detection import Detection
from ..cache.image import ImageContainer
from ..image.formats import ImageFormat

logger = logging.getLogger(__name__)


@register_task('viewer')
class ViewerTask(BaseTask):
    """
    Visualize detections with bounding boxes overlaid on images.
    
    Groups detections by their base image, draws bounding boxes with labels,
    and tiles multiple images horizontally. Saves visualization to /tmp.
    
    Contract:
        Input: IMAGE[label] - Detection objects
        Output: IMAGE[label] - Same Detection objects (pass through)
    
    Usage:
        # Basic: visualize YOLO detections
        yolo() -> viewer()
        
        # Custom label
        yolo(output=my_dets) -> viewer(input=my_dets)
        
        # Configure
        viewer(input=detections, show_children=true)
    """
    
    def __init__(
        self,
        task_id: str = "viewer",
        input_label: str = "detections",
        show_children: bool = True,
        save_dir: str = "/tmp"
    ):
        """
        Initialize viewer task.
        
        Args:
            task_id: Unique task identifier
            input_label: Label to read detections from (default: "detections")
            show_children: Draw child detections recursively (default: True)
            save_dir: Directory to save visualizations (default: "/tmp")
        """
        super().__init__(task_id)
        
        self.input_label = input_label
        self.output_label = input_label  # Pass through to same label
        self.show_children = show_children
        self.save_dir = Path(save_dir)
        
        # Declare contracts
        from ..cache.types import CachedItemType
        self.input_contract = {
            ContextDataType.IMAGE: {
                input_label: (CachedItemType.IMAGE, None)  # Detection objects
            }
        }
        self.output_contract = {
            ContextDataType.IMAGE: {
                self.output_label: (CachedItemType.IMAGE, None)  # Pass through
            }
        }
    
    def configure(self, **kwargs) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            **kwargs: Configuration parameters
                - input: Input label (str)
                - show_children: Show child detections (bool)
                - save_dir: Directory for output files (str)
        """
        if "input" in kwargs:
            self.input_label = kwargs["input"]
            self.output_label = self.input_label
            # Update contracts
            from ..cache.types import CachedItemType
            self.input_contract = {
                ContextDataType.IMAGE: {
                    self.input_label: (CachedItemType.IMAGE, None)
                }
            }
            self.output_contract = {
                ContextDataType.IMAGE: {
                    self.output_label: (CachedItemType.IMAGE, None)
                }
            }
        
        if "show_children" in kwargs:
            show_children_str = str(kwargs["show_children"]).lower()
            self.show_children = show_children_str in ('true', '1', 'yes', 'on')
        
        if "save_dir" in kwargs:
            self.save_dir = Path(kwargs["save_dir"])
    
    def run(self, context: Context) -> Context:
        """
        Visualize detections with bounding boxes.
        
        Args:
            context: Pipeline context with IMAGE[input_label] detections
            
        Returns:
            Context with same detections re-emitted to output_label
        """
        # Get detections from context
        if ContextDataType.IMAGE not in context.data:
            logger.debug(f"No IMAGE data in context")
            return context
        
        if self.input_label not in context.data[ContextDataType.IMAGE]:
            logger.debug(f"No detections in label '{self.input_label}'")
            return context
        
        detections_data = context.data[ContextDataType.IMAGE][self.input_label]
        
        if not detections_data:
            logger.debug(f"Empty detection list in label '{self.input_label}'")
            return context
        
        # Extract Detection objects
        detections: List[Detection] = []
        for item in detections_data:
            if isinstance(item, Detection):
                detections.append(item)
        
        if not detections:
            logger.debug(f"No Detection objects found in '{self.input_label}'")
            return context
        
        # Consume detections (will re-emit after processing)
        context.clear(data_type=ContextDataType.IMAGE, label=self.input_label)
        
        # Group detections by base image
        image_groups = self._group_by_base_image(detections)
        
        # Render each image with its detections
        rendered_images = []
        for base_image, det_list in image_groups.items():
            rendered = self._render_image_with_detections(base_image, det_list)
            rendered_images.append(rendered)
        
        # Tile images horizontally if multiple, otherwise use single image
        if len(rendered_images) > 1:
            final_image = self._tile_images_horizontal(rendered_images)
        else:
            final_image = rendered_images[0]
        
        # Save to file
        output_path = self._save_image(final_image)
        logger.info(f"[{self.task_id}] Visualization saved: {output_path}")
        
        # Re-emit detections to output label (pass through)
        for detection in detections:
            context.add_data(ContextDataType.IMAGE, detection, self.output_label)
        
        return context
    
    def _group_by_base_image(
        self, 
        detections: List[Detection]
    ) -> Dict[ImageContainer, List[Detection]]:
        """
        Group detections by their base image.
        
        Args:
            detections: List of Detection objects
            
        Returns:
            Dictionary mapping base ImageContainer to list of detections
        """
        groups: Dict[ImageContainer, List[Detection]] = {}
        
        for detection in detections:
            base_image = detection.get_base_image()
            if base_image not in groups:
                groups[base_image] = []
            groups[base_image].append(detection)
        
        return groups
    
    def _collect_detections_recursive(
        self,
        det_list: List[Detection],
        depth: int = 0
    ) -> List[tuple]:
        """
        Collect detections with their depth for hierarchical drawing.
        
        Args:
            det_list: List of detections
            depth: Current depth level
            
        Returns:
            List of (detection, depth) tuples
        """
        results = []
        for det in det_list:
            results.append((det, depth))
            
            if self.show_children and det.children:
                results.extend(self._collect_detections_recursive(det.children, depth + 1))
        
        return results
    
    def _render_image_with_detections(
        self,
        base_image: ImageContainer,
        detections: List[Detection]
    ) -> np.ndarray:
        """
        Render base image with detection overlays.
        
        Args:
            base_image: Source ImageContainer
            detections: Detections to draw on this image
            
        Returns:
            Rendered image as numpy array (RGB format from PIL)
        """
        import cv2
        import numpy as np
        
        # Get base image as numpy (RGB from PIL)
        image = base_image.get(ImageFormat.NUMPY).copy()
        
        # Collect all detections (including children)
        all_detections = self._collect_detections_recursive(detections)
        
        # Draw original YOLO detections first (dashed lines in gray)
        # Collect all leaf detections (original YOLO boxes)
        leaf_detections = []
        for detection, depth in all_detections:
            if not detection.children:
                leaf_detections.append(detection)
        
        # Draw original boxes with dashed lines
        for detection in leaf_detections:
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]
            
            # Gray color for original detections
            gray = (128, 128, 128)
            
            # Draw dashed rectangle (by drawing line segments)
            dash_length = 10
            gap_length = 5
            
            # Top edge
            for x in range(x1, x2, dash_length + gap_length):
                cv2.line(image, (x, y1), (min(x + dash_length, x2), y1), gray, 1)
            # Bottom edge
            for x in range(x1, x2, dash_length + gap_length):
                cv2.line(image, (x, y2), (min(x + dash_length, x2), y2), gray, 1)
            # Left edge
            for y in range(y1, y2, dash_length + gap_length):
                cv2.line(image, (x1, y), (x1, min(y + dash_length, y2)), gray, 1)
            # Right edge
            for y in range(y1, y2, dash_length + gap_length):
                cv2.line(image, (x2, y), (x2, min(y + dash_length, y2)), gray, 1)
            
            # Small label for original detection
            label = f"{detection.class_name}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.3
            cv2.putText(image, label, (x1 + 2, y1 + 10), font, font_scale, gray, 1, cv2.LINE_AA)
        
        # Reverse to draw from bottom up (deepest first, top-level last)
        all_detections.reverse()
        
        # Color scheme by depth (RGB format for drawing, will convert to BGR for saving)
        depth_colors = {
            0: (0, 255, 0),      # Green
            1: (255, 255, 0),    # Yellow  
            2: (0, 0, 255),      # Blue (RGB: R=0, G=0, B=255)
            3: (255, 0, 255),    # Magenta
        }
        default_color = (255, 165, 0)  # Orange (RGB)
        
        # Draw cluster detections (solid lines)
        for detection, depth in all_detections:
            x1, y1, x2, y2 = [int(v) for v in detection.bbox]
            
            # Get color based on depth
            color = depth_colors.get(depth, default_color)
            
            # Adjust line thickness based on depth
            thickness = max(1, 3 - depth)
            
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
            
            # Create label
            cluster_id = getattr(detection, 'cluster_id', None)
            if cluster_id is not None:
                label = f"C{cluster_id} "
            else:
                label = ""
            
            if detection.matched_prompts and detection.match_probabilities:
                # Show CLIP match
                label += f"{detection.matched_prompts[0]} {detection.match_probabilities[0]:.3f}"
            else:
                # Show class name and confidence
                label += f"{detection.class_name} {detection.confidence:.2f}"
            
            # Add child count if present
            if detection.children:
                label += f" [{len(detection.children)}]"
            
            # Draw label background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.5 - (depth * 0.05)
            font_scale = max(0.3, font_scale)
            text_thickness = 1
            
            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, text_thickness
            )
            
            # Background rectangle
            cv2.rectangle(
                image,
                (x1, y1 - text_h - baseline - 4),
                (x1 + text_w, y1),
                color,
                -1
            )
            
            # Text
            cv2.putText(
                image,
                label,
                (x1, y1 - baseline - 2),
                font,
                font_scale,
                (0, 0, 0),  # Black text
                text_thickness,
                cv2.LINE_AA
            )
        
        return image
    
    def _tile_images_horizontal(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Tile images horizontally.
        
        Args:
            images: List of numpy arrays (RGB format)
            
        Returns:
            Tiled image as numpy array (RGB)
        """
        import cv2
        
        # Find max height
        max_height = max(img.shape[0] for img in images)
        
        # Resize all images to same height, maintaining aspect ratio
        resized = []
        for img in images:
            h, w = img.shape[:2]
            if h != max_height:
                new_w = int(w * (max_height / h))
                img = cv2.resize(img, (new_w, max_height), interpolation=cv2.INTER_LINEAR)
            resized.append(img)
        
        # Concatenate horizontally
        return np.hstack(resized)
    
    def _save_image(self, image: np.ndarray) -> str:
        """
        Save image to file.
        
        Args:
            image: Image as numpy array (RGB format from PIL)
            
        Returns:
            Path to saved file
        """
        import cv2
        
        # Create output directory if needed
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = int(time.time())
        filename = f"viewer_{self.task_id}_{timestamp}.png"
        output_path = self.save_dir / filename
        
        # Convert RGB to BGR for cv2.imwrite
        bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save image
        cv2.imwrite(str(output_path), bgr_image)
        
        return str(output_path)
    
    def __str__(self) -> str:
        """String representation."""
        return f"ViewerTask(label={self.input_label}, show_children={self.show_children})"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()
