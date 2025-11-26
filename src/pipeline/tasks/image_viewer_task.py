"""
Image viewer task for pipeline integration.

Displays images with optional detection overlays using matplotlib.
Works reliably in background threads and across platforms.
"""

from typing import Optional, Dict, Any
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
import subprocess
import sys
import logging

from ..task_base import BaseTask, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('viewer')
class ImageViewerTask(BaseTask):
    """
    Pipeline task for displaying images with optional detection overlays.
    
    Uses OpenCV to show images in a window. If detections are present
    in the context, they will be overlaid on the image. Works with empty
    detection lists as well.
    
    Accepts:
    - IMAGE (required)
    - DETECTIONS (optional)
    
    Produces: Nothing (sink task - displays to screen)
    """
    
    def __init__(self, task_id: str = "viewer"):
        """
        Initialize image viewer task.
        
        Args:
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.timeout = 3  # Default timeout in seconds (legacy, not used in Agg backend)
        self.window_name = "Image Viewer"
        self.auto_open = True  # Automatically open image in default viewer
        self.show_children = True  # Show child detections recursively (default: True)
        
        # Define contracts - viewer consumes IMAGE and optionally DETECTIONS
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            # DETECTIONS is optional - don't include in contract
        }
        self.output_contract = {}  # Sink task, no outputs
    
    def configure(self, **params) -> None:
        """
        Configure viewer from parameters (DSL support).
        
        Args:
            **params: Keyword arguments with viewer configuration
                - timeout: Display timeout in seconds (legacy, default: 3)
                - window_name: Window title (default: "Image Viewer")
                - auto_open: Whether to automatically open the image (default: True)
                - show_children: Show child detections recursively (default: True)
        
        Example:
            task.configure(timeout=5, window_name="Detection Results", auto_open=True, show_children=False)
        """
        # Parse timeout (legacy parameter)
        if "timeout" in params:
            try:
                self.timeout = float(params["timeout"])
            except (ValueError, TypeError):
                self.timeout = 3
        
        # Store window name if provided
        self.window_name = params.get("window_name", "Image Viewer")
        
        # Store auto_open preference
        if "auto_open" in params:
            auto_open_str = str(params["auto_open"]).lower()
            self.auto_open = auto_open_str in ('true', '1', 'yes', 'on')
        
        # Store show_children preference
        if "show_children" in params:
            show_children_str = str(params["show_children"]).lower()
            self.show_children = show_children_str in ('true', '1', 'yes', 'on')
    
    def run(self, context: Context) -> Context:
        """
        Display image with optional detection overlays.
        
        Args:
            context: Pipeline context containing IMAGE and optionally DETECTIONS
            
        Returns:
            Context (unchanged - sink task)
            
        Raises:
            RuntimeError: If no image in context
        """
        # Get image from context
        if ContextDataType.IMAGE not in context.data or not context.data[ContextDataType.IMAGE]:
            raise RuntimeError(f"Task {self.task_id}: No image in context")
        
        image_data = context.data[ContextDataType.IMAGE][-1]  # Get most recent image
        
        # Handle nested structure from ordered_merge
        if isinstance(image_data, list) and image_data:
            image = image_data[0]  # Unwrap nested list
        else:
            image = image_data
        
        # Convert to numpy array if PIL Image
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image
        
        # Get detections if available
        detections = []
        detection_count = 0
        if ContextDataType.DETECTIONS in context.data and context.data[ContextDataType.DETECTIONS]:
            det_data = context.data[ContextDataType.DETECTIONS]
            
            # Handle ordered_merge nested structure: [[detections], [detections]]
            if isinstance(det_data, list) and len(det_data) > 0 and isinstance(det_data[0], list):
                # Nested - flatten (all branches should have same detections)
                detections = det_data[0]
            elif isinstance(det_data, list):
                # Already flat list
                detections = det_data
            else:
                # Single detection
                detections = [det_data]
            
            detection_count = len(detections)
        
        # Log what we're displaying
        logger.info(f"[{self.task_id}] Displaying image: {image_np.shape}, {detection_count} detections")
        
        # Create figure and axis
        fig, ax = plt.subplots(1, figsize=(10, 8))
        ax.imshow(image_np)
        ax.axis('off')
        ax.set_title(self.window_name)
        
        # Draw detections if present
        # Collect detections to draw (optionally including nested children)
        def collect_detections_recursive(det_list, depth=0):
            """Collect detections with their depth for proper drawing order"""
            results = []
            for det in det_list:
                # Add this detection
                results.append((det, depth))
                # Recursively collect children if show_children is enabled
                if self.show_children and hasattr(det, 'children') and det.children:
                    results.extend(collect_detections_recursive(det.children, depth + 1))
            return results
        
        # Collect all detections with depth information
        all_detections = collect_detections_recursive(detections)
        
        # Define colors for different depths
        depth_colors = {
            0: ('lime', 'lime'),           # Top level: green
            1: ('cyan', 'cyan'),           # Children: cyan
            2: ('yellow', 'yellow'),       # Grandchildren: yellow
            3: ('magenta', 'magenta'),     # Great-grandchildren: magenta
        }
        default_color = ('orange', 'orange')  # Fallback for deeper nesting
        
        # Draw detections (already in bottom-up order from recursion)
        for detection, depth in all_detections:
            # Get bounding box coordinates from box tuple
            x1, y1, x2, y2 = detection.box
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            width = x2 - x1
            height = y2 - y1
            
            # Get colors based on depth
            edge_color, bg_color = depth_colors.get(depth, default_color)
            
            # Adjust linewidth based on depth (thicker for top-level)
            linewidth = 3 - (depth * 0.5)  # 3, 2.5, 2, 1.5, ...
            linewidth = max(1, linewidth)
            
            # Create rectangle patch with depth-based color
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=linewidth, edgecolor=edge_color, facecolor='none'
            )
            ax.add_patch(rect)
            
            # Create label with matched prompts if available
            if hasattr(detection, 'matched_prompts') and detection.matched_prompts and \
               hasattr(detection, 'match_probabilities') and detection.match_probabilities:
                # Show best match
                label = f"{detection.matched_prompts[0]} {detection.match_probabilities[0]:.3f}"
                
                # Add additional matches if present
                if len(detection.matched_prompts) > 1:
                    additional = " | ".join([f"{p} {prob:.3f}" 
                        for p, prob in zip(detection.matched_prompts[1:], detection.match_probabilities[1:])])
                    label += f"\n{additional}"
            else:
                # Fallback to original category
                label = f"{detection.object_category}"
                if hasattr(detection, 'conf') and detection.conf > 0:
                    label += f" {detection.conf:.2f}"
            
            # Add child count indicator for clusters
            if hasattr(detection, 'children') and detection.children:
                label += f" [{len(detection.children)}]"
            
            # Add text with background using depth color
            ax.text(
                x1, y1 - 5,
                label,
                color='black',
                fontsize=10 - (depth * 0.5),  # Slightly smaller text for nested
                weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=bg_color, alpha=0.7, edgecolor='none')
            )
        
        try:
            # Save to file with timestamp in /tmp directory
            import time
            timestamp = int(time.time())
            output_path = f"/tmp/viewer_{self.task_id}_{timestamp}.png"
            
            plt.tight_layout()
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            plt.close(fig)
            
            logger.info(f"[{self.task_id}] Image saved to {output_path}")
            
            # Open the image with default viewer if enabled
            if self.auto_open:
                try:
                    abs_path = os.path.abspath(output_path)
                    if sys.platform == 'darwin':  # macOS
                        # Use plain 'open' to avoid duplicate windows
                        subprocess.Popen(['open', abs_path], 
                                       stdout=subprocess.DEVNULL, 
                                       stderr=subprocess.DEVNULL)
                        logger.info(f"[{self.task_id}] Opened image in Preview")
                    elif sys.platform.startswith('linux'):  # Linux
                        subprocess.Popen(['xdg-open', abs_path],
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL)
                        print(f"[{self.task_id}] Opened image in default viewer")
                    elif sys.platform == 'win32':  # Windows
                        os.startfile(abs_path)
                        print(f"[{self.task_id}] Opened image in default viewer")
                except Exception as e:
                    logger.warning(f"[{self.task_id}] Could not open image automatically: {e}")
                    logger.info(f"[{self.task_id}] Please open manually: {abs_path}")
            else:
                logger.info(f"[{self.task_id}] Auto-open disabled. Open manually: {os.path.abspath(output_path)}")
                    
        except Exception as e:
            logger.error(f"[{self.task_id}] Failed to save image: {e}")
            plt.close(fig)
        
        return context
    
    def describe(self) -> str:
        """Return description of what this task does."""
        return "Displays images in a window with optional detection overlays. Handles empty detection lists gracefully."
    
    def describe_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter descriptions for viewer configuration."""
        return {
            "timeout": {
                "description": "Display timeout in seconds (legacy parameter)",
                "type": "float",
                "default": 3,
                "example": "5.0"
            },
            "window_name": {
                "description": "Window title for the display",
                "type": "str",
                "default": "Image Viewer",
                "example": "Detection Results"
            },
            "auto_open": {
                "description": "Automatically open image in default viewer",
                "type": "bool",
                "default": True,
                "example": "true"
            }
        }
