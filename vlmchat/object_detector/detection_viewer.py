"""
ObjectDetector pipeline stage for visualizing detections.

This class chains to a source detector and uses an ImageViewer
to display the source image with bounding boxes drawn for
all detections.
"""

from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
import cv2 # For color conversion

from .detection_base import ObjectDetector, Detection
from .image_viewer import (
    ImageViewer, COLOR_GREEN, COLOR_BLUE, COLOR_RED, COLOR_YELLOW,
    COLOR_ORANGE, COLOR_PURPLE, COLOR_WHITE, COLOR_BLACK
)



class DetectionViewer(ObjectDetector):
    """
    An ObjectDetector that visualizes detections from its source.
    
    This class is a pipeline 'sink' or 'tap'. It draws all detections
    it receives from its source onto the image and displays it using
    an ImageViewer. It then passes the detections on.
    """
    
    def __init__(self, *, 
                 source: ObjectDetector, 
                 viewer: ImageViewer,
                 display_time_ms: int = 1,
                 show_children: bool = True):
        """
        Initializes the DetectionViewer.
        
        Args:
            source: The preceding detector in the pipeline (required).
            viewer: An initialized ImageViewer instance to display images.
            display_time_ms: Time to display each frame in milliseconds.
                           - 0: Don't display (skip visualization)
                           - -1: Wait for keypress
                           - >0: Display for specified milliseconds (default: 1)
            show_children: Whether to recursively draw child detections (default: True).
                          If False, only top-level detections are drawn.
        """
        super().__init__(source)
        self._viewer = viewer
        self._ready = False
        self._display_time_ms = display_time_ms
        self._show_children = show_children
        
        # Color palette for different object categories
        self._colors = [
            COLOR_GREEN, COLOR_BLUE, COLOR_RED, COLOR_YELLOW,
            COLOR_ORANGE, COLOR_PURPLE
        ]
        self._color_map = {}

    def start(self) -> None:
        """Starts the viewer and its source."""
        super().start()
        if not self._viewer.is_visible():
            print("Warning: ImageViewer window is not visible.")
        self._ready = True
        print("DetectionViewer started.")

    def stop(self) -> None:
        """Stops the viewer and its source."""
        super().stop()
        self._ready = False
        # We don't close the viewer, as it's managed externally
        print("DetectionViewer stopped.")

    def readiness(self) -> bool:
        """
        Checks if this detector, its source, and the viewer are ready.
        """
        source_ready = super().readiness()
        return self._ready and self._viewer.is_visible() and source_ready

    def get_labels(self) -> List[str]:
        """Passes through the labels from the source."""
        return super().get_labels()

    def _get_color_for_category(self, category: str) -> Tuple[int, int, int]:
        """Gets a consistent color for a given object category."""
        if category not in self._color_map:
            # Assign a new color from the palette
            next_color_index = len(self._color_map) % len(self._colors)
            self._color_map[category] = self._colors[next_color_index]
        return self._color_map[category]

    def _draw_detection_recursive(self, vis_image: np.ndarray, det: Detection, depth: int = 0):
        """
        Recursively draw a detection and all its children with different colors based on depth.
        
        Args:
            vis_image: Image to draw on
            det: Detection to draw
            depth: Current depth level (0 = parent, 1 = child, 2 = grandchild, etc.)
            
        Returns:
            Image with the detection and its children drawn
        """
        # Cycle through colors based on depth
        color_idx = depth % len(self._colors)
        color = self._colors[color_idx]
        
        # Draw box for this detection
        x1, y1, x2, y2 = det.box
        
        # Build label with matched prompts if available
        if hasattr(det, 'matched_prompts') and det.matched_prompts and \
           hasattr(det, 'match_probabilities') and det.match_probabilities:
            # Create multi-line label: main label + additional matches
            labels = []
            
            # Main label with best match
            main_label = f"[ID {det.id}] {det.matched_prompts[0]} {det.match_probabilities[0]:.3f}"
            if depth > 0:
                main_label += f" [d:{depth}]"
            if hasattr(det, 'children') and len(det.children) > 0:
                main_label += f" [{len(det.children)} children]"
            labels.append(main_label)
            
            # Additional matches as separate lines
            for p, prob in zip(det.matched_prompts[1:], det.match_probabilities[1:]):
                labels.append(f"  + {p} {prob:.3f}")
            
            label = labels
        else:
            # Fallback to original category and confidence
            label = f"[ID {det.id}] {det.object_category} {det.conf:.2f}"
            
            # Add depth indicator
            if depth > 0:
                label += f" [d:{depth}]"
            
            # Add child count if present
            if hasattr(det, 'children') and len(det.children) > 0:
                label += f" [{len(det.children)} children]"
        
        # Vary thickness based on depth (thicker = higher level)
        thickness = max(1, 3 - depth)
        
        # Draw box (returns modified image)
        vis_image = self._viewer.draw_box(
            vis_image,
            (int(x1), int(y1), int(x2), int(y2)),
            color=color,
            thickness=thickness,
            label=label
        )
        
        # Recursively draw children if enabled
        if self._show_children and hasattr(det, 'children'):
            for child in det.children:
                vis_image = self._draw_detection_recursive(vis_image, child, depth + 1)
        
        return vis_image
    
    def _detect_internal(self, image: Image, detections: List[Detection]) -> List[Detection]:
        """
        Visualizes the detections from the source detector.
        
        Args:
            image: The input PIL Image
            detections: The list of detections from the source detector
            
        Returns:
            The same list of detections (pass-through)
        """
        # Skip visualization if display_time_ms is 0
        if self._display_time_ms == 0:
            return detections
        
        # Convert PIL Image to numpy array for drawing
        if isinstance(image, Image.Image):
            vis_image = np.array(image)
        else:
            vis_image = image.copy()

        # Draw each detection recursively (this will draw children as well)
        for det in detections:
            vis_image = self._draw_detection_recursive(vis_image, det, depth=0)

        # Convert RGB to BGR for OpenCV display (PIL images are RGB, OpenCV expects BGR)
        if isinstance(image, Image.Image):
            vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)

        # Display the image using ImageViewer
        # -1 means wait for keypress (cv2.waitKey(0))
        # >0 means wait for specified milliseconds
        wait_time = 0 if self._display_time_ms == -1 else self._display_time_ms
        self._viewer.show(vis_image, wait_ms=wait_time)

        return detections


# Example usage:
if __name__ == "__main__":
    
    # --- Import dependencies for example ---
    from yolo_detector_cpu import YoloV8Detector, PersonFilter # Use filter from yolo
    import time
    import requests # For downloading image
    from io import BytesIO # For reading image from bytes
    
    if 'YoloV8Detector' not in globals() or 'ImageViewer' not in globals():
        print("\nCannot run example: Missing dependencies (YOLO or ImageViewer).")
        print("Please ensure yolo_detector.py and image_viewer.py are available.")
    else:
        print("\n--- Detection Viewer Pipeline Example ---")
        
        # 1. Load a real image from URL
        try:
            image_url = "https://encyclopediaofalabama.org/wp-content/uploads/2023/04/Trail-Riders-1300x903.jpg"
            print(f"Loading image from URL: {image_url}")
            response = requests.get(image_url)
            response.raise_for_status() # Raise an error for bad responses
            pil_image = Image.open(BytesIO(response.content))
            print(f"Loaded image successfully (Size: {pil_image.size})")

        except Exception as e:
            print(f"Error loading image from URL: {e}")
            print("Using dummy image instead. Real detections will not occur.")
            width, height = 640, 480
            img_array = np.ones((height, width, 3), dtype=np.uint8) * 128
            pil_image = Image.fromarray(img_array)

        # 2. Initialize the viewer
        viewer = ImageViewer(window_name="Detection Pipeline")
        
        # 3. Create the 3-stage pipeline:
        # Stage 1: YoloV8Detector (Produces detections)
        # Stage 2: PersonFilter (Filters detections)
        # Stage 3: DetectionViewer (Visualizes remaining detections)
        print("Creating pipeline: YOLO -> PersonFilter -> DetectionViewer")
        detector1 = YoloV8Detector(model_name='yolov8n.pt')
        #detector2 = PersonFilter(source=detector1)
        detector3 = DetectionViewer(source=detector1, viewer=viewer)
        
        try:
            # 4. Start the pipeline (starts all detectors)
            detector3.start()
            
            if detector3.readiness():
                # 5. Run detection
                print("\nRunning detection pipeline...")
                # This will run YOLO, then filter, then display
                detections = detector3.detect(pil_image)
                
                print(f"\nPipeline finished. {len(detections)} detections were visualized.")
                
                # 6. Keep window open for 5 seconds
                print("Displaying result for 5 seconds...")
                start_time = time.time()
                while time.time() - start_time < 5 and viewer.is_visible():
                    viewer.show(wait_ms=30) # Keep processing GUI events
            
            else:
                print("Pipeline failed to start.")
                
        except Exception as e:
            print(f"An error occurred: {e}")
        finally:
            # 7. Stop the pipeline and close the viewer
            print("Stopping pipeline...")
            detector3.stop()
            viewer.close()
            print("--- End of Example ---")