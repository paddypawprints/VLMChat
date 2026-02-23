# Device MQTT Message Validation

## Overview

The camera-framework includes strict MQTT message validation to ensure security and data integrity. All incoming MQTT commands are validated against JSON schemas before processing.

## Security Model

- **Whitelist-based**: Only known topics are accepted
- **Fail-fast**: Device terminates on unknown topic or validation failure
- **Schema enforcement**: All messages must conform to their JSON schema

## Validation Behavior

### Startup
- Validator loads all required schemas on initialization
- Device **terminates immediately** if:
  - Schemas directory not found
  - Required schema files missing
  - Schema files contain invalid JSON
  - Schema files are not valid JSON Schema

### Runtime
- All incoming MQTT messages are validated before handlers are called
- Device **terminates immediately** if:
  - Message received on unknown topic (security violation)
  - Message payload fails schema validation
  - Unexpected validation errors occur

### Allowed Topics

The device only accepts messages on these topics:
- `devices/{device_id}/commands/metrics` - Metrics configuration (start/stop, frequency)
- `devices/{device_id}/commands/snapshot` - Snapshot capture command

Any message on a different topic causes immediate termination.

## Configuration

### Schemas Path

By default, the validator looks for schemas at `../../shared/schemas` relative to the camera-framework installation.

You can override this using:

**Command Line:**
```bash
python -m macos_device --mqtt --schemas-path /path/to/schemas
```

**Environment Variable:**
```bash
export SCHEMAS_PATH=/path/to/schemas
python -m macos_device --mqtt
```

**Programmatic:**
```python
from camera_framework.bridges import DeviceClient
from camera_framework.mqtt_validator import MQTTValidator

# Pass to DeviceClient
client = DeviceClient(
    device_id="my-device",
    device_type="camera",
    runner=runner,
    broker_host="localhost",
    schemas_path="/path/to/schemas"
)
```

### Schema Directory Structure

The schemas directory must follow this structure:
```
schemas/
├── metrics-config/
│   └── v1.0.0/
│       └── schema.json
└── command-simple/
    └── v1.0.0/
        └── schema.json
```

## Production Deployment

For production deployments:

1. **Include schemas in package**: Copy the `/shared/schemas` directory into your deployment package
2. **Set SCHEMAS_PATH**: Configure the environment variable or CLI argument to point to the bundled schemas
3. **Monitor logs**: Watch for validation errors - they indicate potential attacks or client bugs

## Example Validation Flow

```
1. MQTT message arrives on: devices/mac-dev-01/commands/metrics
2. Validator checks topic against whitelist ✓
3. Validator loads schema: metrics-config/v1.0.0/schema.json
4. Validator validates payload against schema ✓
5. Handler is called with validated payload
```

## Error Examples

### Unknown Topic (Security Violation)
```
CRITICAL: Unknown MQTT topic received: devices/mac-dev-01/commands/unauthorized
CRITICAL: This is a security violation - device terminating
```
**Device terminates immediately**

### Invalid Payload
```
CRITICAL: Invalid message received on devices/mac-dev-01/commands/metrics
CRITICAL: Validation error: 'enabled' is a required property
CRITICAL: Failed at path: 
CRITICAL: Payload: {"frequency": 60}
CRITICAL: Device terminating due to validation failure
```
**Device terminates immediately**

### Missing Schema at Startup
```
CRITICAL: Required schema not found: /schemas/metrics-config/v1.0.0/schema.json
```
**Device terminates immediately**

## Rationale

This strict validation approach provides:

1. **Security**: Unknown topics are rejected immediately, preventing unauthorized commands
2. **Data Integrity**: Invalid messages are caught before processing, preventing crashes or corruption
3. **Fail-Fast**: Problems are detected immediately rather than causing subtle bugs
4. **Defense in Depth**: Validation occurs on both server and device sides

The receiver-validates pattern means:
- Server validates messages from devices (register, heartbeat, status, etc.)
- Browser validates messages from server (snapshots, metrics, etc.)
- **Device validates messages from server (commands)** ← This implementation

## Testing

To test validation:

```python
# Test with valid command
mosquitto_pub -t "devices/mac-dev-01/commands/metrics" \
  -m '{"enabled": true, "frequency": 60}'

# Test with invalid command (missing required field)
mosquitto_pub -t "devices/mac-dev-01/commands/metrics" \
  -m '{"frequency": 60}'  # Device will terminate

# Test with unknown topic
mosquitto_pub -t "devices/mac-dev-01/commands/hack" \
  -m '{"evil": true}'  # Device will terminate
```

## Troubleshooting

### "Schemas directory not found"
- Check SCHEMAS_PATH is set correctly
- Verify directory exists: `ls -la $SCHEMAS_PATH`
- Default path is relative to camera-framework installation

### "Required schema not found"
- Check schema files exist in expected locations
- Verify directory structure matches expected format
- Check file permissions (must be readable)

### "Invalid JSON Schema"
- Validate schema file is valid JSON: `jq . schema.json`
- Validate schema conforms to JSON Schema spec
- Check for syntax errors in schema file

## See Also

- [AsyncAPI MQTT Specification](/shared/specs/asyncapi-mqtt.yaml)
- [Shared Schemas Directory](/shared/schemas/)
- [Server-Side MQTT Validation](/web-platform/server/mqtt-validator.ts)
