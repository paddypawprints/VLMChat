# YOLOv8n Model

YOLOv8 nano model wrapper with TensorRT runtime support for efficient object detection.

## Overview

This module provides a clean interface to YOLOv8n object detection through the model abstraction layer. It follows the same pattern as other models in the system (CLIP, SmolVLM, etc.) with swappable runtime backends.

## Architecture

- **YoloModel**: Main facade class that manages runtimes and provides the detection API
- **YoloRuntimeBase**: Abstract interface defining what YOLO backends must implement
- **YoloTensorRTBackend**: Concrete implementation using TensorRT for optimized inference

## Usage

```python
from models.Yolov8n import YoloModel
from utils.config import VLMChatConfig
import cv2

# Load configuration
config = VLMChatConfig.from_json("config.json")

# Create model (automatically uses TensorRT runtime)
yolo = YoloModel(config)

# Load image
image = cv2.imread("image.jpg")

# Perform detection
detections = yolo.detect(
    image,
    confidence_threshold=0.25,
    iou_threshold=0.45
)

# Process results
for det in detections:
    bbox = det["bbox"]  # [x1, y1, x2, y2]
    confidence = det["confidence"]
    class_name = det["class_name"]
    print(f"{class_name}: {confidence:.2f} at {bbox}")
```

## Configuration

The model uses the TensorRT engine path from the configuration file:

```json
{
  "model": {
    "yolo_engine_path": "~/Dev/model-rt-build/platform/jetson/release_artifacts/yolov8n_fp16.engine"
  }
}
```

## Detection Output

Each detection is a dictionary with:
- `bbox`: List of [x1, y1, x2, y2] coordinates in original image space
- `confidence`: Float confidence score (0-1)
- `class_id`: Integer class ID (0-79 for COCO)
- `class_name`: String class name (e.g., "person", "car")

## Relationship to Detector

This model provides the inference engine for YOLO detection. The detector layer (`src/object_detector/yolo_detector.py`) can use this model, but also supports other use cases like the IMX500 camera's on-device YOLO inference.

## TensorRT Requirements

- TensorRT installed
- PyCUDA installed
- YOLOv8n engine file generated and configured
- CUDA-capable GPU

## Class Names

The model loads COCO class names from `src/object_detector/coco_names.json` if available, otherwise falls back to numeric labels.
