# Camera Framework

Lightweight pipeline framework for edge device AI vision applications.

## Design Philosophy

**Context + Runner = The Framework**

Everything else is tasks. Keep it minimal.

## Installation

```bash
# Install in editable mode
pip install -e .
```

## Quick Start

```python
from camera_framework import Runner, Context, BaseTask, Collector
from camera_framework.metrics import RateInstrument

class Camera(BaseTask):
    def process(self, ctx: Context):
        # Capture frame
        frame = capture_frame()
        
        # Store in context (list semantics)
        ctx.append("frame", frame)

class Display(BaseTask):
    def process(self, ctx: Context):
        # Get latest frame
        frame = ctx.get("frame", -1)  # -1 = last item
        show(frame)

# Create pipeline
runner = Runner([Camera(), Display()])

# Add metrics
collector = Collector()
collector.add_instrument(RateInstrument("fps"), "frame")

# Run
while True:
    collector.record("frame", 1.0)
    runner.run_once()
    
    if collector.get_stats("fps")["count"] % 30 == 0:
        print(f"FPS: {collector.get_stats('fps')['rate']:.1f}")
```

## Core Components

### RefCountedCache

Thread-safe automatic memory management for pipeline data.

```python
from camera_framework.cache import RefCountedCache

cache = RefCountedCache()

# Store data
key = cache.incref(data)

# Retrieve
data = cache.get(key)

# Release (auto-cleanup when refcount reaches 0)
cache.decref(key)
```

### Context

List-based storage for pipeline data using native Python reference counting.

```python
from camera_framework import Context

ctx = Context()

# List operations
ctx.append("frame", frame_key)
ctx.extend("objects", [obj1, obj2, obj3])

# Access
latest = ctx.get("frame", -1)  # Last item
first_five = ctx.get("objects", slice(0, 5))  # Slice

# Modify
ctx.set("frame", 0, new_key)  # Update index
ctx.delete("frame", slice(0, 10))  # Delete range
ctx.clear("frame")  # Clear all

# Copy (increfs all keys)
ctx2 = ctx.copy()

# Release (decrefs all keys)
ctx.release()
```

### BaseTask

Abstract base class for pipeline tasks.

```python
from camera_framework import BaseTask, Context

class MyTask(BaseTask):
    def __init__(self, fields=None):
        super().__init__(name="my_task", fields=fields)
    
    def process(self, ctx: Context):
        # Use self.field() for field name mapping
        input_key = ctx.get(self.field("input"), 0)
        
        # Process...
        output = do_work(input_key)
        
        # Store result
        ctx.append(self.field("output"), output)
```

**Field Mapping:**

```python
# Remap field names
task = MyTask(fields={"input": "frame", "output": "processed"})

# task.field("input") returns "frame"
# task.field("output") returns "processed"
# task.field("unknown") returns "unknown" (passthrough)
```

### Runner

Sequential task execution with automatic context management.

```python
from camera_framework import Runner

runner = Runner([task1, task2, task3], collector=metrics)

# Run once (creates context, runs tasks, releases)
runner.run_once()

# Get runner statistics
stats = runner.stats()
# {'tasks': 15}
```

### Metrics

Observer pattern for performance tracking.

```python
from camera_framework import Collector
from camera_framework.metrics import (
    RateInstrument,
    AvgDurationInstrument,
    CounterInstrument,
    RecentSamplesInstrument,
)

collector = Collector("device-01")

# Add instruments
collector.add_instrument(RateInstrument("fps"), "frame")
collector.add_instrument(AvgDurationInstrument("latency"), "latency")
collector.add_instrument(CounterInstrument("detections"), "detection")
collector.add_instrument(RecentSamplesInstrument("recent", max_samples=100), "value")

# Record data
collector.record("frame", 1.0)
collector.record("detection", 5)
collector.record("value", 42.0)

# Time operations
with collector.duration_timer("latency"):
    do_expensive_work()

# Get statistics
stats = collector.get_stats()
# {
#   'fps': {'rate': 29.8, 'count': 1000, ...},
#   'latency': {'avg': 0.033, 'count': 1000, ...},
#   'detections': {'total': 5000, ...},
#   'recent': {'samples': [42.0, ...], ...}
# }

# Get specific instrument
fps_stats = collector.get_stats("fps")
```

**Available Instruments:**

- `RateInstrument` - Events per second (sliding window)
- `AvgDurationInstrument` - Average duration in seconds
- `CounterInstrument` - Cumulative counter
- `RecentSamplesInstrument` - Recent N samples

**NullCollector:**

```python
from camera_framework.metrics import null_collector

# Disable metrics
runner = Runner(tasks, collector=null_collector)
```

## MQTT Integration

Device communication via MQTT bridges.

```python
from camera_framework.bridges import DeviceClient

client = DeviceClient(
    device_id="jetson-01",
    device_type="jetson",
    broker_host="mqtt.example.com",
    broker_port=1883,
)

# Add tasks
client.add_tasks([camera, yolo, display])

# Start (connects MQTT, registers device)
client.start()

# Run (main loop with metrics publishing)
try:
    client.run()
except KeyboardInterrupt:
    client.stop()
```

See [BRIDGES.md](BRIDGES.md) for detailed MQTT documentation.

## Testing

```bash
python -m pytest tests/test_smoke.py -v
```

6 smoke tests covering:
- Cache lifecycle and refcounting
- Context copy/release
- Basic pipeline execution
- Field mapping
- Metrics collection
- Context list operations

## Examples

See `macos-device/` for complete example with camera and display tasks.

```bash
# Standalone mode
python -m macos_device

# MQTT mode
python -m macos_device --mqtt
```

## Design Decisions

### Why List Semantics?

Tasks like YOLO produce multiple detections per frame. Context needs to handle:
- Multiple objects per field
- Batch processing
- Temporal sequences

Lists are the natural fit.

### Why Reference Counting?

Edge devices (Jetson with 8GB RAM) can't afford data copies. Reference counting:
- Tracks object lifetimes automatically
- Enables safe sharing between tasks
- Prevents memory leaks
- Zero-copy when possible

### Why No Session in Metrics?

The old metrics had a Session abstraction between Collector and Instruments. This added complexity without benefit. Instruments now observe Collector directly, reducing from 900 to 200 lines.

### Why Field Mapping?

Tasks need to be reusable with different field names:

```python
# Use same Display task for different inputs
display_raw = Display(fields={"frame": "raw_frame"})
display_processed = Display(fields={"frame": "processed_frame"})
```

Dictionary-based mapping is simple and explicit.

## Dependencies

- `numpy==1.24.2` - Pinned for transformers compatibility
- `paho-mqtt>=1.6.0` - MQTT client (optional, bridges only)
- `cryptography>=41.0.0` - PKI authentication (optional, bridges only)
- `psutil>=5.9.0` - System specs (optional, bridges only)

## License

See LICENSE file.

## Documentation

- [BRIDGES.md](BRIDGES.md) - MQTT integration guide
- [STATUS.md](STATUS.md) - Current implementation status
- [ARCHITECTURE.md](ARCHITECTURE.md) - Design decisions and patterns
