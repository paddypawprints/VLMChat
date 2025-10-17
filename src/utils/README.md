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

### Camera Architecture
Abstract base class architecture supporting multiple camera models and platforms.

**Files Structure:**
- `camera_base.py` - Base camera interface with enums and BaseCamera abstract class
- `detection_base.py` - Generic Detection class and ObjectDetectionInterface
- `imx500_camera.py` - Basic IMX500Camera class (no detection)
- `imx500_detection.py` - IMX500ObjectDetection class (inherits from both interfaces)
- `camera_factory.py` - CameraFactory for creating camera instances

**Key Design Features:**
- **Generic Detection class** - No IMX500 dependencies, uses integer coordinates
- **Multiple inheritance** - IMX500ObjectDetection inherits from both interfaces
- **Factory pattern** - `CameraFactory.create_camera()` with support for detection flag
- **Extensible** - Ready for IMX477/IMX219 and Jetson platform support
- **Clean separation** - Detection logic separate from basic camera functionality

**Usage:**
```python
from utils.camera_factory import CameraFactory
from utils.camera_base import CameraModel, Platform, Device

# Basic camera
camera = CameraFactory.create_camera(CameraModel.IMX500)

# Camera with detection
detection_camera = CameraFactory.create_camera(
    CameraModel.IMX500,
    with_detection=True,
    args=args
)

# Convenience method for detection
detection_camera = CameraFactory.create_detection_camera(CameraModel.IMX500)
```

**Supported Hardware:**
- **Camera Models**: IMX500 (IMX477, IMX219 planned)
- **Platforms**: Raspberry Pi (Jetson planned)
- **Devices**: camera0, camera1

### camera.py (Legacy)
Original IMX500 camera interface - use camera_factory.py for new implementations.

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

## Metrics (quick reference)

The repository includes a lightweight metrics system at `src/utils/metrics_collector.py`.
It is useful for in-process telemetry and quick instrumenting of operations.

Examples

```py
from src.utils.metrics_collector import Collector, Session, ValueType
from src.utils.metrics_collector import CounterInstrument

collector = Collector()
collector.register_timeseries("requests", registered_attribute_keys=["route"], max_count=256)

session = Session(collector)
counter = CounterInstrument("reqs", binding_keys=["route"])
session.add_instrument(counter, "requests")

collector.data_point("requests", {"route": "/home"}, 1)

with collector.duration_timer("handler.latency", attributes={"route": "/home"}):
    handle_request()

path = session.export_to_json("/tmp/metrics")
print("Wrote session export:", path)
```

See `tests/test_metrics_collector.py` and `tests/test_metrics_instruments.py` for more examples and behavior expectations.