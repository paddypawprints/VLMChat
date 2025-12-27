# Edge AI Device SDK

Python SDK for edge devices to communicate with the VLMChat platform.

**This package is auto-generated from the platform's shared schemas and specifications.**

## Installation

```bash
pip install -e .
```

## Usage

### MQTT Communication

```python
from edge_llm_client.models.alerts import Alert
from edge_llm_client.models.metrics_data import MetricsData
import paho.mqtt.client as mqtt
from datetime import datetime
import json

# Connect to MQTT broker
client = mqtt.Client()
client.connect("localhost", 1883, 60)

# Publish an alert
alert = Alert(
    type="detection",
    timestamp=datetime.now().isoformat(),
    description="Person detected",
    confidence=0.92
)

client.publish(
    f"devices/{device_id}/alerts",
    alert.model_dump_json(),
    qos=1
)

# Publish metrics
metrics = MetricsData(
    session="vision_pipeline",
    timestamp=datetime.now().isoformat(),
    instruments=[
        {
            "name": "fps",
            "type": "gauge",
            "value": 30.5
        }
    ]
)

client.publish(
    f"devices/{device_id}/metrics/data/vision_pipeline",
    metrics.model_dump_json(),
    qos=1
)
```

### Device Registration

```python
from edge_llm_client.models.device_register import DeviceRegister

registration = DeviceRegister(
    name="RaspberryPi-4B-Living-Room",
    type="raspberry-pi",
    ip="192.168.1.100",
    specs={
        "cpu": "Cortex-A72",
        "memory": "4GB",
        "camera": "IMX500"
    }
)

client.publish(
    f"devices/{device_id}/register",
    registration.model_dump_json(),
    qos=1
)
```

## Development

This SDK is generated from:
- `shared/schemas/` - JSON Schema definitions
- `shared/specs/asyncapi.yaml` - MQTT topic specifications

To regenerate:
```bash
cd /path/to/VLMChat
just generate-python
```

## Models

All Pydantic models are in `edge_llm_client.models/`:
- `alerts` - Detection and system alerts
- `device_register` - Device registration
- `metrics_config` - Metrics configuration
- `metrics_data` - Performance metrics
- `watchlist` - Detection watchlist
- `webrtc_signaling` - WebRTC signaling
