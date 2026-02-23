# MQTT Bridges

The `camera_framework.bridges` module provides MQTT integration for device applications.

## Architecture

Bridges are split by MQTT topic/responsibility:

- **mqtt_client.py** (~150 lines) - Core MQTT transport layer
- **registration.py** (~180 lines) - Device lifecycle (PKI, registration, heartbeat)
- **metrics_publisher.py** (~120 lines) - Telemetry publishing with filtering
- **device_client.py** (~140 lines) - Orchestrator that ties everything together

Total: ~590 lines (vs 700-line monolith)

## Usage

### Standalone Mode (No MQTT)

```python
from camera_framework import Runner, Collector
from macos_device import Camera, Display

runner = Runner([Camera(), Display()])
collector = Collector()

while True:
    runner.run_once()
```

### MQTT Mode (DeviceClient)

```python
from camera_framework.bridges import DeviceClient
from macos_device import Camera, Display

client = DeviceClient(
    device_id="mac-dev-01",
    device_type="macos",
    broker_host="localhost",
    broker_port=1883,
)

client.add_tasks([Camera(), Display()])

client.start()  # Connect MQTT, register device
client.run()    # Main loop with metrics publishing
client.stop()   # Cleanup
```

## MQTT Topics

### Device → Backend

- `devices/{id}/register` - Device registration with PKI public key and specs
- `devices/{id}/heartbeat` - Periodic heartbeat (30s interval)
- `devices/{id}/metrics` - Metrics data (frequency configurable via command)
- `devices/{id}/status` - Online/offline status

### Backend → Device

- `devices/{id}/commands/metrics` - Configure metrics publishing
  ```json
  {
    "enabled": true,
    "frequency": 60,
    "instruments_filter": ["pipeline.fps", "camera.frames"]
  }
  ```

## PKI Authentication

Devices use Ed25519 keypairs for authentication:

- Private key: `~/.vlmchat_device_key_{id}.pem`
- Public key: Included in registration message
- Auto-generated if not found

## Device Specs

Registration includes hardware information:

```json
{
  "device_id": "mac-dev-01",
  "device_type": "macos",
  "public_key": "<base64-encoded-public-key>",
  "specs": {
    "cpu_model": "Apple M1",
    "cpu_count": 8,
    "memory_total_mb": 16384,
    "temperature": 45.2,
    "python_version": "3.11.13"
  }
}
```

## Metrics Filtering

Configure which instruments to publish:

```python
# Backend sends command:
{
  "enabled": true,
  "frequency": 60,  # seconds
  "instruments_filter": ["pipeline.fps", "yolo.duration"]
}
```

Metrics publisher only sends matching instruments, reducing bandwidth.

## Example: macOS Device

```bash
# Standalone mode (no MQTT)
python -m macos_device

# MQTT mode
python -m macos_device --mqtt
```

## Dependencies

- `paho-mqtt>=1.6.0` - MQTT client
- `cryptography>=41.0.0` - Ed25519 PKI
- `psutil>=5.9.0` - System specs (CPU, memory, temperature)

## Design Decisions

1. **Topic-based split** - Each module handles related MQTT topics
2. **Thread safety** - Background threads for heartbeat and metrics publishing
3. **Auto-reconnect** - MQTT client handles reconnection automatically (5s timeout)
4. **Filtering** - Metrics publisher supports selective instrument publishing
5. **PKI** - Ed25519 keys provide authentication without passwords

## Migration from device_app.py

Old monolith (700 lines) → New bridges (590 lines across 4 files):

- Clearer separation of concerns
- Easier to test individual components
- Reusable across macos-device and jetson-device platforms
- Same functionality, better organization
