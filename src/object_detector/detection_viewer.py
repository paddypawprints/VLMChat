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

from detection_base import ObjectDetector, Detection
from image_viewer import (
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
                 viewer: ImageViewer):
        """
        Initializes the DetectionViewer.
        
        Args:
            source: The preceding detector in the pipeline (required).
            viewer: An initialized ImageViewer instance to display images.
        """
        super().__init__(source)
        self._viewer = viewer
        self._ready = False
        
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

    def _detect_internal(self, image: Image, detections: List[Detection]) -> List[Detection]:
        """
        Draws detections on the image and displays it.
        
        This method does not modify the detection list itself,
        it just visualizes the results.
        """
        
        # 1. Convert PIL Image (RGB) to OpenCV Image (BGR)
        # np.array creates a copy
        cv_image_rgb = np.array(image)
        cv_image_bgr = cv2.cvtColor(cv_image_rgb, cv2.COLOR_RGB2BGR)
        
        display_image = cv_image_bgr
        
        # 2. Draw all detections on the image
        for det in detections:
            color = self._get_color_for_category(det.object_category)
            labels = [
                det.object_category,
                f"{det.conf:.2f}"
            ]
            
            display_image = self._viewer.draw_box(
                image=display_image,
                box=det.box,
                color=color,
                label=labels
            )
            
        # 3. Show the image with all boxes
        self._viewer.show(display_image)
        
        # 4. Return the original, unmodified detection list
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