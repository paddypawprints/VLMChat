# Environment Design - Dictionary-based Key-Value Store

## Overview

The Environment singleton has been refactored to use a dictionary-based key-value store instead of hardcoded attributes. This makes it **generally usable by any task** without requiring modifications to the Environment class itself.

## Key Format

The recommended key format is:
```
{taskType}+{taskId}+{key}
```

### Examples:
- `Camera+cam1+current_image` - Camera task instance "cam1" stores its current image
- `ObjectDetector+det1+results` - Object detector instance "det1" stores detection results
- `App+main+history` - Chat application stores conversation history
- `ImageProcessor+proc1+threshold` - Image processor stores its threshold parameter

## Why This Design?

### ✅ Advantages

1. **Fully Extensible**: Any task can store/retrieve data without modifying Environment
2. **Namespace Separation**: Multiple instances of the same task type can coexist
3. **No Coupling**: Tasks don't need to know about each other's data structures
4. **Easy Debugging**: `env.keys()` shows all active data
5. **General Purpose**: Works for any data type (images, configs, results, etc.)

### 🎯 Use Cases

- **Pipeline Tasks**: Share intermediate results between tasks
- **State Management**: Store task configuration and runtime state
- **Cross-component Communication**: Chat app and pipeline system share data
- **Multi-instance Tasks**: Multiple cameras, detectors, etc. with separate state

## API Reference

### Core Methods

```python
from pipeline.environment import Environment

# Get singleton instance
env = Environment.get_instance()

# Set a value
env.set("Camera+cam1+current_image", image)

# Get a value
image = env.get("Camera+cam1+current_image")

# Get with default
image = env.get("Camera+cam1+current_image", default_image)

# Check if key exists
if env.has("Camera+cam1+current_image"):
    image = env.get("Camera+cam1+current_image")

# Alternative syntax using 'in'
if "Camera+cam1+current_image" in env:
    image = env.get("Camera+cam1+current_image")

# Remove a key
removed = env.remove("Camera+cam1+current_image")  # Returns True if removed

# Get all keys
all_keys = env.keys()

# Get number of keys
count = len(env)

# Clear all data
env.clear()

# Reset singleton (mainly for testing)
Environment.reset()
```

### Backward Compatibility

For existing code that used the old attribute-based API:

```python
# Old style - still works via properties
env.current_image = image
env.history = history_manager

# Or via methods
env.set_image(image)
env.get_image()
env.clear_image()

# These map to:
# "App+main+current_image"
# "App+main+history"
```

## Usage Examples

### Example 1: Camera Task Storing Image

```python
from pipeline.environment import Environment

class CameraTask:
    def __init__(self, task_id="cam1"):
        self.task_id = task_id
        self.env = Environment.get_instance()
    
    def capture(self):
        image = self.hardware_capture()
        key = f"Camera+{self.task_id}+current_image"
        self.env.set(key, image)
        return image
```

### Example 2: Detector Using Camera Image

```python
from pipeline.environment import Environment

class ObjectDetectorTask:
    def __init__(self, task_id="det1", camera_id="cam1"):
        self.task_id = task_id
        self.camera_id = camera_id
        self.env = Environment.get_instance()
    
    def execute(self):
        # Get image from camera task
        image_key = f"Camera+{self.camera_id}+current_image"
        image = self.env.get(image_key)
        
        if image is None:
            raise ValueError(f"No image available from camera {self.camera_id}")
        
        # Run detection
        results = self.detect_objects(image)
        
        # Store results for next task
        results_key = f"ObjectDetector+{self.task_id}+results"
        self.env.set(results_key, results)
        
        return results
```

### Example 3: Display Task Using Detector Results

```python
from pipeline.environment import Environment

class DisplayTask:
    def __init__(self, detector_id="det1"):
        self.detector_id = detector_id
        self.env = Environment.get_instance()
    
    def execute(self):
        # Get detection results
        results_key = f"ObjectDetector+{self.detector_id}+results"
        
        if not self.env.has(results_key):
            print(f"No results available from detector {self.detector_id}")
            return
        
        results = self.env.get(results_key)
        self.display_detections(results)
```

### Example 4: Multiple Task Instances

```python
from pipeline.environment import Environment

# Create multiple camera instances
env = Environment.get_instance()

# Camera 1
env.set("Camera+cam1+current_image", image1)
env.set("Camera+cam1+resolution", (1920, 1080))

# Camera 2
env.set("Camera+cam2+current_image", image2)
env.set("Camera+cam2+resolution", (640, 480))

# Each maintains separate state
cam1_image = env.get("Camera+cam1+current_image")
cam2_image = env.get("Camera+cam2+current_image")
```

### Example 5: Configuration Storage

```python
from pipeline.environment import Environment

class ConfigurableTask:
    def __init__(self, task_id):
        self.task_id = task_id
        self.env = Environment.get_instance()
        
        # Store default configuration
        config_key = f"{self.__class__.__name__}+{task_id}+config"
        self.env.set(config_key, {
            "threshold": 0.5,
            "max_items": 10,
            "enabled": True
        })
    
    def get_config(self):
        config_key = f"{self.__class__.__name__}+{self.task_id}+config"
        return self.env.get(config_key, {})
    
    def update_config(self, updates):
        config = self.get_config()
        config.update(updates)
        config_key = f"{self.__class__.__name__}+{self.task_id}+config"
        self.env.set(config_key, config)
```

## Best Practices

### 1. Use Consistent Key Format
```python
# Good - consistent format
key = f"{self.__class__.__name__}+{self.task_id}+{data_name}"

# Also good - explicit task type
key = f"Camera+{self.task_id}+current_image"
```

### 2. Check Before Getting
```python
# Good - check existence first
if env.has(key):
    value = env.get(key)

# Or use default
value = env.get(key, default_value)
```

### 3. Clean Up When Done
```python
# Remove data when no longer needed
env.remove(f"Task+{task_id}+temp_data")

# Or clear all task data
for key in env.keys():
    if key.startswith(f"Task+{task_id}+"):
        env.remove(key)
```

### 4. Document Your Keys
```python
class MyTask:
    """
    Task that processes images.
    
    Environment Keys Used:
    - Camera+cam1+current_image (input): PIL.Image
    - MyTask+{task_id}+results (output): List[Dict]
    - MyTask+{task_id}+config (state): Dict
    """
```

## Migration from Old API

If you have existing code using the old attribute-based API:

### Before (Hardcoded Attributes)
```python
env = Environment.get_instance()
env.current_image = image
env.history = history

# Access
image = env.current_image
history = env.history
```

### After (Dictionary-based)
```python
env = Environment.get_instance()
env.set("App+main+current_image", image)
env.set("App+main+history", history)

# Access
image = env.get("App+main+current_image")
history = env.get("App+main+history")
```

### Backward Compatibility
The old API still works via properties:
```python
# This still works!
env.current_image = image
image = env.current_image

# Maps to: env.set("App+main+current_image", image)
```

## Debugging Tips

### List All Active Keys
```python
env = Environment.get_instance()
print("Active environment keys:")
for key in env.keys():
    value = env.get(key)
    print(f"  {key}: {type(value).__name__}")
```

### Check Data Flow
```python
# At task boundaries, log what's being stored/retrieved
logger.debug(f"Storing result: {key}")
env.set(key, result)

logger.debug(f"Retrieving input: {key}")
input_data = env.get(key)
```

### Clear Between Pipeline Runs
```python
# Clear task-specific data between runs
def cleanup_task_data(task_id):
    env = Environment.get_instance()
    prefix = f"Task+{task_id}+"
    for key in list(env.keys()):  # Copy list to avoid modification during iteration
        if key.startswith(prefix):
            env.remove(key)
```

## Summary

The refactored Environment provides a **general-purpose, extensible key-value store** that:
- ✅ Works for any task type without code changes
- ✅ Supports multiple instances of the same task
- ✅ Maintains backward compatibility
- ✅ Enables clean namespace separation
- ✅ Makes debugging easier with `keys()` inspection

This design scales naturally as you add new task types and use cases!
