# macos-device

macOS development/testing device for the camera pipeline. Runs a full YOLO + attribute detection + MQTT publishing pipeline using the `camera-framework`.

## Prerequisites

- Python 3.10+
- `camera-framework` installed (see `camera-framework/`)
- YOLOv8n weights (`yolov8n.pt`) in the project root
- PA100K attribute ONNX model (path configured in `macos_device_config.yaml`)
- Config files in the project root:
  - `camera_framework_config.yaml`
  - `macos_device_config.yaml`

## Installation

From the project root:

```bash
pip install -e camera-framework/
pip install -e macos-device/
```

## Configuration

Two YAML config files (both read from the **working directory**, i.e. the project root):

**`camera_framework_config.yaml`** — pipeline/framework settings:
- `pipeline.max_workers` — thread pool size
- `pipeline.memory_leak_threshold` — seconds before a held object is flagged
- `sources.image_library.image_dir` — path to image dataset (e.g. MOT15)
- `sources.image_library.framerate` — simulated FPS

**`macos_device_config.yaml`** — device-specific settings:
- `tasks.yolo.model_path` — path to YOLO weights
- `tasks.attributes.model_path` — path to PA100K ONNX model
- `tasks.color_filter.*` — color region and matching thresholds
- `sinks.mqtt.*` — MQTT broker address, topic prefix, schemas path

## Running

Always run from the **project root** so the config files are found:

```bash
# Standalone mode — camera + YOLO only, no MQTT
python -m macos_device

# Full pipeline with MQTT integration
python -m macos_device --mqtt

# Full pipeline with explicit schemas path
python -m macos_device --mqtt --schemas-path ./shared/schemas

# Generate a Mermaid pipeline diagram and exit
python -m macos_device --diagram
```

### Modes

| Mode | Flag | Description |
|------|------|-------------|
| Standalone | _(none)_ | Camera → YOLO only. No MQTT. Useful for testing the detector. |
| MQTT | `--mqtt` | Full pipeline: Camera → YOLO → Attributes → Clustering → Tracking → Alerts → MQTT |
| Diagram | `--diagram` | Builds the full pipeline, dumps a Mermaid diagram to stdout, and exits |

## Pipeline (MQTT mode)

```
Camera → YoloDetector → AttributeEnricher → YoloCategoryRouter
       → AttributeColorFilter → Clusterer → DetectionTracker
       → DetectionFilter → SmolVLMVerifier → AlertPublisher → MQTT
```

The pipeline runs in a tight loop (`runner.run_once()` + 10ms sleep). Stats are logged every 30 frames showing FPS, average frame duration, cache size, and memory leak warnings.

## Stopping

Press `Ctrl+C`. The pipeline performs a clean shutdown: tasks are stopped in reverse order, the MQTT client disconnects, and the thread pool is drained.

## Tests

```bash
cd macos-device
pytest tests/
```
