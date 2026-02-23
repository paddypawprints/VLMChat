# Camera Framework - Current Status

## Overview

Minimal camera-framework v2 with MQTT device integration.

**Total Lines: ~1,030**
- camera-framework core: ~440 lines
- MQTT bridges: ~590 lines

## Structure

```
camera-framework/
├── camera_framework/
│   ├── __init__.py          # Main exports
│   ├── cache.py             # RefCountedCache (~60 lines)
│   ├── context.py           # Context with list semantics (~120 lines)
│   ├── task.py              # BaseTask with field mapping (~30 lines)
│   ├── runner.py            # Runner with cache (~30 lines)
│   ├── metrics.py           # Collector + 4 instruments (~200 lines)
│   └── bridges/
│       ├── __init__.py      # Bridge exports
│       ├── mqtt_client.py   # MQTT transport (~150 lines)
│       ├── registration.py  # Device lifecycle (~180 lines)
│       ├── metrics_publisher.py # Telemetry (~120 lines)
│       └── device_client.py # Orchestrator (~140 lines)
├── tests/
│   └── test_smoke.py        # 6 smoke tests (all passing)
├── pyproject.toml           # Dependencies
├── BRIDGES.md               # MQTT integration guide
└── README.md

macos-device/
├── macos_device/
│   ├── __main__.py          # Standalone + MQTT modes
│   ├── camera.py            # cv2.VideoCapture
│   └── display.py           # cv2.imshow
└── pyproject.toml
```

## Core Concepts

### 1. Reference-Counted Cache

```python
cache = RefCountedCache()
key = cache.incref(data)   # Store data, get key
data = cache.get(key)      # Retrieve data
cache.decref(key)          # Release reference (auto-cleanup at 0)
```

### 2. Context (List Semantics)

```python
ctx = Context(cache)
ctx.append("frame", frame_key)        # Append to list
ctx.extend("objects", [obj1, obj2])   # Extend list
frame = ctx.get("frame", 0)           # Get by index
frames = ctx.get("frame", slice(0, 5)) # Get slice
```

### 3. Field Mapping

```python
class Display(BaseTask):
    def __init__(self, fields=None):
        super().__init__(name="display", fields=fields)
    
    def process(self, ctx):
        # self.field("frame") returns "image" if fields={"frame": "image"}
        frame_key = ctx.get(self.field("frame"), 0)
```

### 4. Runner

```python
runner = Runner([camera, yolo, display], collector=metrics)
runner.run_once()  # Create context, run tasks, release (auto-cleanup)
```

### 5. Metrics

```python
collector = Collector("device")
collector.add_instrument(RateInstrument("fps"), "frame")
collector.add_instrument(AvgDurationInstrument("duration"), "duration")

collector.record("frame", 1.0)
with collector.duration_timer("duration"):
    runner.run_once()

stats = collector.get_stats()  # Get all instrument stats
```

### 6. MQTT Integration

```python
from camera_framework.bridges import DeviceClient

client = DeviceClient(
    device_id="mac-01",
    device_type="macos",
    broker_host="localhost",
)

client.add_tasks([camera, display])
client.start()  # Connect, register
client.run()    # Main loop with metrics
```

## Key Features

✅ **Minimal Core** - ~440 lines for framework primitives
✅ **Automatic Memory Management** - Reference counting prevents leaks
✅ **List Semantics** - Context stores lists, not single values
✅ **Field Mapping** - Tasks can remap field names via dictionary
✅ **Simple Metrics** - Observer pattern, no Session layer
✅ **MQTT Bridges** - Modular device integration (~590 lines)
✅ **PKI Authentication** - Ed25519 keypairs for secure device auth
✅ **Platform Agnostic** - Core framework has no platform dependencies

## Design Principles

1. **Context + Runner = The Framework** - Everything else is tasks
2. **Platform-specific tasks stay in device projects** - macos-device, jetson-device
3. **Smoke tests only** - High-value, not high-coverage
4. **YAGNI** - "Do I need this NOW?" test for features
5. **Modular MQTT** - Split by topic/responsibility (~100-180 lines each)

## Dependencies

**camera-framework:**
- `numpy==1.24.2` (transformers compatibility)
- `paho-mqtt>=1.6.0` (MQTT client)
- `cryptography>=41.0.0` (Ed25519 PKI)
- `psutil>=5.9.0` (system specs)

**macos-device:**
- `opencv-python` (camera/display)
- `camera-framework` (local editable install)

## Testing

```bash
cd camera-framework
python -m pytest tests/test_smoke.py -v
```

**6 smoke tests, all passing in 0.18s:**
- test_cache_lifecycle
- test_context_refcounting
- test_basic_pipeline
- test_field_mapping
- test_metrics_basic
- test_context_list_operations

## Usage Examples

### Standalone Mode (Development)

```bash
cd macos-device
python -m macos_device
```

Shows FPS and frame time, displays camera feed.

### MQTT Mode (Production)

```bash
cd macos-device
python -m macos_device --mqtt
```

Connects to MQTT broker, registers device, publishes metrics.

## Next Steps

1. **Add YOLO task** to macos-device (ultralytics)
2. **Add Clusterer task** to macos-device
3. **Test full pipeline**: Camera→YOLO→Clusterer→Display
4. **Verify metrics publishing** to MQTT broker
5. **Create jetson-device** structure (mirror macos-device)
6. **Test on Jetson hardware** with GStreamer camera

## Migration Notes

**From old vlmchat pipeline:**
- Removed DSL completely
- Removed Session abstraction from metrics
- Simplified from 900 to 200 lines (metrics)
- Split 700-line device_app.py into 4 focused modules

**What stayed:**
- Reference-counted cache concept
- Observer pattern for metrics
- MQTT topics and message formats
- PKI authentication approach

**What changed:**
- Context now stores lists, not single values
- Field mapping via dictionary instead of DSL
- No Session in metrics (instruments observe Collector directly)
- MQTT split by topic/responsibility
