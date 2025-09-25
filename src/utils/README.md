# Utilities Module

This module contains utility functions and classes that provide common functionality across the VLMChat application, including image processing and camera interface.

## Components

### image_utils.py
Utility functions for loading and processing images from various sources.

**Key Functions:**
- `load_image_from_url()`: Download and load images from URLs
- `load_image_from_file()`: Load images from local file paths

**Features:**
- **Format Standardization**: Converts all images to RGB format
- **Error Handling**: Comprehensive error logging and graceful failures
- **Network Support**: HTTP/HTTPS image downloading with streaming
- **Format Support**: Handles various image formats via PIL

**Usage:**
```python
from utils.image_utils import load_image_from_url, load_image_from_file

# Load from URL
image = load_image_from_url("https://example.com/image.jpg")
if image:
    print(f"Loaded image: {image.size}")

# Load from local file
image = load_image_from_file("/path/to/image.png")
if image:
    print(f"Loaded image: {image.size}")
```

### camera.py
IMX500 camera interface for Raspberry Pi with object detection capabilities.

**Key Classes:**
- `Detection`: Represents a single object detection result
- `IMX500ObjectDetection`: Main camera interface with neural processing

**Features:**
- **Hardware Integration**: Direct integration with Raspberry Pi IMX500 camera
- **Object Detection**: Built-in neural network inference
- **Image Capture**: Single image capture with timestamp naming
- **Real-time Processing**: Continuous detection loop with async processing
- **Configurable Parameters**: Adjustable detection thresholds and limits

**Usage:**
```python
from utils.camera import IMX500ObjectDetection

# Initialize camera
camera = IMX500ObjectDetection()

# Capture single image
filepath, image = camera.capture_single_image()
print(f"Image saved to: {filepath}")

# Run continuous detection (blocking)
camera.run_detection_loop()
```

## Image Processing

### Format Standardization
All images are converted to RGB format for consistent processing:
- Handles RGBA, grayscale, and other formats
- Ensures compatibility with vision models
- Maintains color information for analysis

### Error Handling
Comprehensive error handling for common issues:
- Network connectivity problems
- Invalid image formats
- File access permissions
- Corrupted image data

### Performance Optimization
- Streaming download for large images
- Memory-efficient image processing
- Lazy loading where appropriate

## Camera Integration

### IMX500 Features
- **Neural Processing**: On-device inference with built-in NPU
- **Object Detection**: Real-time object recognition
- **High Performance**: Hardware-accelerated image processing
- **Low Latency**: Minimal delay from capture to processing

### Detection Capabilities
The camera supports object detection with:
- Configurable confidence thresholds
- Non-maximum suppression (NMS)
- Multiple object categories (COCO labels)
- Bounding box coordinate conversion

### Configuration Options
```python
# Camera configuration parameters
args = IMX500ObjectDetection.get_args()
# --model: Path to neural network model
# --threshold: Confidence threshold (default: 0.55)
# --iou: IoU threshold for NMS (default: 0.65)
# --max-detections: Maximum detections per frame (default: 10)
# --fps: Target frame rate
```

## Usage Patterns

### Basic Image Loading
```python
from utils.image_utils import load_image_from_url

# Simple URL loading
image = load_image_from_url("https://example.com/photo.jpg")
if image:
    # Process image
    print(f"Image size: {image.size}")
    print(f"Image mode: {image.mode}")  # Should be 'RGB'
```

### Camera Capture Integration
```python
from utils.camera import IMX500ObjectDetection

# Initialize and capture
camera = IMX500ObjectDetection()
filepath, image = camera.capture_single_image()

# Image is ready for vision model processing
# filepath contains the saved file location
# image is a PIL Image object in RGB format
```

### Error-Resilient Loading
```python
from utils.image_utils import load_image_from_file

# Attempt to load with error handling
image = load_image_from_file("potentially_missing_file.jpg")
if image is None:
    print("Failed to load image, using default")
    # Handle failure case
else:
    # Process successfully loaded image
    pass
```

## Dependencies

### image_utils.py
- **PIL (Pillow)**: Image processing and format conversion
- **Requests**: HTTP client for URL downloads
- **IO**: Byte stream handling for image data
- **Logging**: Error reporting and debugging

### camera.py
- **Picamera2**: Raspberry Pi camera interface
- **NumPy**: Array processing for neural network data
- **PIL**: Image format conversion
- **Multiprocessing**: Async detection processing
- **Threading**: Concurrent frame processing

## Hardware Requirements

### For Camera Module
- Raspberry Pi 4 or newer
- IMX500 camera module with neural processing unit
- Sufficient power supply for neural inference
- Compatible ribbon cable connection

### For Image Utils
- Standard Python environment
- Internet connectivity for URL loading
- File system access for local images

This utilities module provides the foundation for image handling and camera integration throughout the VLMChat application.