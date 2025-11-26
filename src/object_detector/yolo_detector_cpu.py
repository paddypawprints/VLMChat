"""
Implementation of an ObjectDetector using Ultralytics YOLOv8.
"""

from typing import List, Optional, Tuple
from PIL import Image
import numpy as np
import logging

logger = logging.getLogger(__name__)

# --- Dependencies for YOLO implementation ---
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

# --- Import Base Classes ---
from .detection_base import ObjectDetector, Detection


# --- YOLOv8 Implementation ---

class YoloV8Detector(ObjectDetector):
    """
    ObjectDetector implementation using Ultralytics YOLOv8.
    
    This class loads a YOLOv8 model and performs inference.
    It acts as a *producer* in a pipeline, adding new detections to
    any detections passed from its source.
    """
    
    def __init__(self, *, 
                 model_name: str = 'yolov8n.pt', 
                 source: Optional[ObjectDetector] = None):
        """
        Initializes the YOLOv8 detector.

        Args:
            model_name: The name of the YOLOv8 model to load (e.g., 'yolov8n.pt').
            source: An optional preceding detector in the pipeline.
        """
        super().__init__(source) # Call base __init__
        
        if YOLO is None:
            raise ImportError("Ultralytics YOLO package is not installed.")
            
        self.model_name = model_name
        self.model: Optional[YOLO] = None
        self._labels: List[str] = []
        self._ready: bool = False

    def start(self, audit: bool = False) -> None:
        """
        Loads the YOLOv8 model and prepares it for inference.
        
        Args:
            audit: Optional audit flag passed to source detector
        """
        super().start(audit) # Start the source, if any
        
        if self._ready: # Already started
            return
            
        try:
            logger.debug(f"Loading model {self.model_name} for CPU inference...")
            self.model = YOLO(self.model_name)
            
            # Perform a dummy inference to warm up the model
            dummy_image = Image.fromarray(np.uint8(np.zeros((640, 640, 3))))
            self.model(dummy_image, device='cpu', verbose=False)
            
            # Store labels
            if self.model and self.model.names:
                # model.names is a dict like {0: 'person', 1: 'bicycle', ...}
                self._labels = [self.model.names[i] for i in range(len(self.model.names))]
            
            self._ready = True
            logger.debug(f"Model {self.model_name} loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading YOLOv8 model: {e}")
            self.model = None
            self._ready = False

    def stop(self) -> None:
        """
        Cleans up and unloads the model.
        """
        super().stop() # Stop the source, if any
        self.model = None
        self._ready = False
        self._labels = []
        logger.debug(f"YOLOv8 detector ({self.model_name}) stopped.")

    def readiness(self) -> bool:
        """
        Checks if this model is loaded AND its source (if any) is ready.
        """
        source_ready = self._source.readiness() if self._source else True
        return self._ready and self.model is not None and source_ready

    def get_labels(self) -> List[str]:
        """
        Returns the list of class labels from this detector AND its source,
        with duplicates removed.
        """
        source_labels = super().get_labels()
        # Combine labels, avoiding duplicates
        combined_labels = source_labels + [l for l in self._labels if l not in source_labels]
        return combined_labels

    def _detect_internal(self, image: Image, detections: List[Detection]) -> List[Detection]:
        """
        Performs object detection and adds results to the detection list.

        Args:
            image: The input PIL Image.
            detections: The list of detections from the source (if any).

        Returns:
            The list of detections, now including detections from this model.
        """
        if not self._ready or self.model is None:
            # Model isn't ready, just return the detections from the source
            return detections

        try:
            # Perform inference. verbose=False silences console output.
            results = self.model(image, device='cpu', verbose=False)
            
            # Check for results and process them
            if results and results[0]:
                if hasattr(results[0], 'boxes'):
                    boxes = results[0].boxes
                    if boxes is not None and len(boxes) > 0:
                        
                        for box in boxes:
                            # Extract bounding box
                            xyxy = box.xyxy[0].cpu().numpy().astype(int)
                            box_tuple: Tuple[int, int, int, int] = (xyxy[0], xyxy[1], xyxy[2], xyxy[3])
                            
                            # Extract confidence
                            confidence = box.conf[0].cpu().item()
                            
                            # Extract class ID and name
                            class_id = int(box.cls[0].cpu().item())
                            class_name = self.model.names[class_id] if self.model.names else f"class_{class_id}"
                            
                            # Create the Detection object
                            detection = Detection(
                                box=box_tuple,
                                object_category=class_name,
                                conf=confidence
                            )
                            detections.append(detection) # Add to the list
                    
        except Exception as e:
            print(f"Error during YOLOv8 detection: {e}")

        return detections # Return the modified list


# --- Define a simple filter detector (Moved to module level) ---
class PersonFilter(ObjectDetector):
    """A simple detector that filters for 'person'."""
    
    def __init__(self, source: ObjectDetector):
        super().__init__(source)
        self._ready = False
    
    def start(self) -> None:
        super().start()
        self._ready = True
        print("PersonFilter started.")

    def stop(self) -> None:
        super().stop()
        self._ready = False
        print("PersonFilter stopped.")

    def readiness(self) -> bool:
        # Our readiness is just our source's readiness
        return super().readiness()

    def get_labels(self) -> List[str]:
        # A filter doesn't add labels, it just passes them through
        return super().get_labels() 

    def _detect_internal(self, image: Image, detections: List[Detection]) -> List[Detection]:
        # Filter the list from the source
        print(f"PersonFilter: Received {len(detections)} detections.")
        filtered_detections = [
            d for d in detections if d.object_category == 'person'
        ]
        print(f"PersonFilter: Returning {len(filtered_detections)} 'person' detections.")
        return filtered_detections


# Example usage:
if __name__ == "__main__":
    
    if YOLO is None:
        print("Cannot run example: 'ultralytics' is not installed.")
    elif 'ObjectDetector' in globals() and 'ObjectDetector' in globals() and globals()['ObjectDetector'] is None:
         print("Cannot run example: 'object_detector_base.py' not found.")
    else:
        print("--- YOLOv8 Detector Pipeline Example ---")
        from utils.image_utils import load_image_from_url
        # 1. Load a real image (or use dummy)
        try:
            # *** IMPORTANT ***
            # For a real test, change this path to an image with people
            # e.g., "path/to/my_image.jpg"
            image_path = "https://images.pdimagearchive.org/collections/berg-and-hoeg/32856644182_d379f3512b_o.jpg?width=1140&height=800"
            pil_image = load_image_from_url(image_path)
            print(f"Loaded image from: {image_path}")
        except FileNotFoundError:
            print("Using dummy image. Real detections will not occur.")
            width, height = 640, 480
            img_array = np.ones((height, width, 3), dtype=np.uint8) * 128
            pil_image = Image.fromarray(img_array)

        # 2. Create the pipeline
        print("\nCreating pipeline: YoloV8Detector -> PersonFilter")
        detector1 = YoloV8Detector()
        detector2 = PersonFilter(source=detector1) # detector2 wraps detector1
        
        # 3. Start the pipeline (this will start detector1, then detector2)
        detector2.start()

        if detector2.readiness():
            # 4. Get labels
            labels = detector2.get_labels()
            print(f"\nPipeline Labels (first 10): {labels[:10]}")

            # 5. Perform detection
            print(f"\nDetecting objects in image...")
            detections = detector2.detect(pil_image)

            if detections:
                print(f"\n--- Final Detections ({len(detections)}) ---")
                for det in detections:
                    print(f"  - {det}")
            else:
                print("\nNo final detections found.")

            # 6. Stop the detector
            detector2.stop()
        else:
            print("Detector failed to start. Exiting example.")
            
        print("--- End of Example ---")