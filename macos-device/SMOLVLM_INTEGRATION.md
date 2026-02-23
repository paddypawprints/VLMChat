# SmolVLM Pipeline Integration

## Architecture

```
Camera → YOLO → CategoryRouter → PersonAttributes → AttributeColorFilter 
  → Clusterer → DetectionTracker → VLMQueue → [AlertPublisher, SmolVLMWorker]
                                        ↓
                                   SmolVLMWorker → MQTT (VLM-verified alerts)
```

## VLMQueue Routing Logic

**Smart routing based on `vlm_required` flag and SmolVLM worker state:**

1. **vlm_required=False** → Route to AlertPublisher immediately
2. **vlm_required=True + SmolVLM ready** → Route to SmolVLMWorker
3. **vlm_required=True + SmolVLM busy** → Queue message
4. **Queue full** → Drop oldest, send to AlertPublisher with `vlm_timeout=True`

## Buffer Flow

```
DetectionTracker
       ↓
   tracker_buffer (size=10)
       ↓
    VLMQueue
    ↙      ↘
alert_buffer  smolvlm_buffer (size=1)
    ↓              ↓
AlertPublisher  SmolVLMWorker
                   ↓
                 MQTT (vlm-verified alerts)
```

## Event Fields

**From DetectionTracker:**
- `filter_id`, `track_id`, `crop_jpeg`, `bbox`
- `attributes`, `colors`, `confidence`
- `confirmation_count`, `first_seen`, `timestamp`
- `vlm_required` (bool) - From SearchFilter.vlm_required
- `vlm_reasoning` (str) - From SearchFilter.vlm_reasoning

**Added by VLMQueue:**
- `queued_at` (timestamp) - When queued
- `vlm_timeout` (bool) - If dropped due to full queue

**Added by SmolVLMWorker:**
- `vlm_verified` (bool) - True if processed
- `vlm_result` (str) - Generated text from SmolVLM
- `vlm_inference_time` (float) - Seconds for inference
- `vlm_error` (str) - Error message if failed

## Configuration

Add to `macos_device_config.yaml`:

```yaml
tasks:
  smolvlm:
    model_path: "~/onnx/SmolVLM2-256M-Instruct"
    model_size: "256M"  # or "500M"
    max_new_tokens: 100
    queue_size: 10
```

## Usage Example

```python
from .vlm_queue import VLMQueue
from .smolvlm_worker import SmolVLMWorker

# Create buffers
tracker_buffer = Buffer(size=10, policy=drop_oldest_policy, name="tracker")
alert_buffer = Buffer(size=10, policy=drop_oldest_policy, name="alerts")
smolvlm_buffer = Buffer(size=1, policy=drop_oldest_policy, name="smolvlm")

# Create VLM worker
vlm_worker = SmolVLMWorker(
    model_path="~/onnx/SmolVLM2-256M-Instruct",
    model_size="256M",
    device_id="mac-dev-01",
    mqtt_client=mqtt_client,
)

# Create VLM queue with worker reference
vlm_queue = VLMQueue(
    max_queue_size=10,
    smolvlm_worker=vlm_worker,
)

# Wire pipeline
tracker.outputs.append(tracker_buffer)
vlm_queue.inputs.append(tracker_buffer)
vlm_queue.add_outputs(alert_buffer, smolvlm_buffer)
vlm_worker.inputs.append(smolvlm_buffer)

# Create alert publisher
alert_publisher = AlertPublisher(
    device_id="mac-dev-01",
    mqtt_client=mqtt_client,
)
alert_publisher.inputs.append(alert_buffer)

# Add to runner
runner.add_task(tracker)
runner.add_task(vlm_queue)
runner.add_task(vlm_worker)
runner.add_task(alert_publisher)
```

## Performance

**256M Model (Jetson Orin Nano 8GB):**
- Memory: ~3.5GB GPU
- Inference: 5-15 seconds per detection
- Queue prevents pipeline stalls
- Immediate alerts for non-VLM detections

**Queue Behavior:**
- Max queue size: 10 detections
- If full: Drop oldest, emit with `vlm_timeout=True`
- VLMQueue checks worker every iteration
- SmolVLMWorker signals busy/ready state

## Metrics

**VLMQueue:**
```python
stats = vlm_queue.get_stats()
# {
#   'routed_to_alerts': 45,
#   'routed_to_vlm': 12,
#   'queued': 8,
#   'timeouts': 2,
#   'queue_full_drops': 2,
#   'queue_size': 3,
#   'queue_capacity': 10,
#   'smolvlm_ready': False
# }
```

**SmolVLMWorker:**
```python
stats = vlm_worker.get_stats()
# {
#   'detections_processed': 10,
#   'vlm_successes': 9,
#   'vlm_failures': 1,
#   'total_inference_time': 123.4,
#   'avg_inference_time': 12.34,
#   'is_busy': True,
#   'models_loaded': True
# }
```
