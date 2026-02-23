# Configuration Management API

This feature provides RESTful endpoints for managing device and framework configurations with versioning and audit history.

## Features

- **Device Configurations**: Per-device task and sink settings
- **Framework Configurations**: Shared pipeline settings (sources, buffers, etc.)
- **Versioning**: Automatic version incrementing with optimistic locking
- **History**: Full audit trail of all configuration changes
- **Rollback**: Restore previous configurations
- **MQTT Sync**: Automatically push config updates to devices

## API Endpoints

### Device Configuration

#### Get Device Config
```
GET /api/config/device/:deviceId
```

Returns the active configuration for a device.

**Response:**
```json
{
  "id": "uuid",
  "deviceId": "device-id",
  "config": { ... },
  "version": 3,
  "isActive": true,
  "updatedBy": "user-id",
  "createdAt": "2026-01-21T...",
  "updatedAt": "2026-01-21T..."
}
```

#### Update Device Config
```
PUT /api/config/device/:deviceId
```

Creates a new version of the device configuration.

**Request:**
```json
{
  "config": {
    "tasks": {
      "yolo": { ... },
      "attributes": { ... }
    },
    "sinks": {
      "mqtt": { ... }
    }
  },
  "changeDescription": "Updated YOLO confidence threshold"
}
```

**Response:** Returns the new config record with incremented version.

#### Get Config History
```
GET /api/config/device/:deviceId/history?limit=10
```

Returns configuration history for a device.

#### Rollback Config
```
POST /api/config/device/:deviceId/rollback/:historyId
```

Restores a previous configuration version.

### Framework Configuration

#### Get Framework Config
```
GET /api/config/framework
```

Returns the active framework configuration (shared pipeline settings).

#### Update Framework Config
```
PUT /api/config/framework
```

**Request:**
```json
{
  "name": "default",
  "config": {
    "pipeline": {
      "max_workers": 4,
      "memory_leak_threshold": 30.0
    },
    "sources": { ... },
    "buffers": { ... }
  },
  "changeDescription": "Increased worker count"
}
```

#### Get Framework Config History
```
GET /api/config/framework/history?limit=10
```

Returns framework configuration history.

## Database Schema

### deviceConfigs
- `id`: UUID primary key
- `deviceId`: Foreign key to devices table (unique)
- `config`: JSONB - full device configuration
- `version`: Integer - incremented on each update
- `isActive`: Boolean - only one active config per device
- `updatedBy`: Foreign key to users table
- `createdAt`, `updatedAt`: Timestamps

### frameworkConfigs
- `id`: UUID primary key
- `name`: String - config name (default: "default")
- `config`: JSONB - full framework configuration
- `version`: Integer - incremented on each update
- `isActive`: Boolean - only one active config at a time
- `updatedBy`: Foreign key to users table
- `createdAt`, `updatedAt`: Timestamps

### configHistory
- `id`: UUID primary key
- `configType`: 'device' | 'framework'
- `configId`: Reference to deviceConfigs.id or frameworkConfigs.id
- `deviceId`: Foreign key to devices (NULL for framework configs)
- `config`: JSONB - snapshot of config at this version
- `version`: Integer - version number
- `changeDescription`: String - optional description
- `changedBy`: Foreign key to users table
- `createdAt`: Timestamp

## MQTT Integration

When a device configuration is updated, the system automatically publishes the new config to the device:

```
Topic: devices/{deviceId}/commands/config
Payload: {
  "version": 3,
  "config": { ... }
}
```

The device should:
1. Validate the new config
2. Apply the configuration
3. Respond with acknowledgment on `devices/{deviceId}/status`

## Security

- All endpoints require authentication via `requireAuth` middleware
- Device configs: User must own the device
- Framework configs: Any authenticated user can read/update (consider adding admin role)
- Config history is immutable (no DELETE operations)

## Example: Device Config Structure

```yaml
tasks:
  yolo:
    model_path: "yolov8n.pt"
    confidence: 0.25
    iou: 0.45
    device: "cpu"
  
  attributes:
    model_path: "/path/to/pa_model.onnx"
    confidence_threshold: 0.5
    batch_size: 1
  
  color_filter:
    regions: { ... }
    matching: { ... }
    hsv: { ... }
  
  clusterer:
    max_clusters: 10
    merge_threshold: 0.6
    weights: { ... }
  
  tracker:
    confirmation: { ... }
    matching: { ... }

sinks:
  mqtt:
    broker_host: "localhost"
    broker_port: 1883
    device_id: "device-123"
```

## Example: Framework Config Structure

```yaml
pipeline:
  max_workers: 4
  memory_leak_threshold: 30.0
  stats_interval: 30

sources:
  camera:
    width: 1920
    height: 1080
    fps: 30.0
    device: 0

buffers:
  default_size: 5
  alert_size: 10
  policy: "drop_oldest"
```

## Testing

```bash
# Get device config
curl -H "Authorization: Bearer $SESSION_ID" \
  http://localhost:3000/api/config/device/mac-dev-01

# Update device config
curl -X PUT -H "Authorization: Bearer $SESSION_ID" \
  -H "Content-Type: application/json" \
  -d '{"config": {...}, "changeDescription": "Test update"}' \
  http://localhost:3000/api/config/device/mac-dev-01

# Get history
curl -H "Authorization: Bearer $SESSION_ID" \
  http://localhost:3000/api/config/device/mac-dev-01/history

# Rollback
curl -X POST -H "Authorization: Bearer $SESSION_ID" \
  http://localhost:3000/api/config/device/mac-dev-01/rollback/$HISTORY_ID
```
