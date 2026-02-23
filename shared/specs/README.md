# API Documentation Overview

This platform uses **three separate API specifications** for different protocols:

## 1. REST API - OpenAPI Spec
**File**: [`shared/specs/openapi.yaml`](openapi.yaml)  
**Protocol**: HTTP/REST  
**Client**: Web browser (admin UI)  
**Server**: Express.js REST API

### Endpoints
- **Authentication**: `/api/auth/*`
- **Device Management**: `/api/devices/*`
- **Chat Interface**: `/api/chat/*`
- **Schemas**: `/api/schemas/*`
- **Metrics**: `/api/devices/{deviceId}/metrics/*`

### Schema References
```yaml
# Uses shared schemas
Device: 
  allOf:
    - $ref: '../schemas/register/v1.0.0/schema.json'
    - additionalProperties...

MetricsConfig:
  $ref: '../schemas/metrics-config/v1.0.0/schema.json'

MetricsData:
  $ref: '../schemas/metrics-data/v1.0.0/schema.json'
```

## 2. MQTT API - AsyncAPI Spec  
**File**: [`shared/specs/asyncapi-mqtt.yaml`](asyncapi-mqtt.yaml)  
**Protocol**: MQTT  
**Client**: Edge devices (Python)  
**Server**: Mosquitto MQTT broker

### Topics
- **Device → Server**:
  - `devices/{deviceId}/register` - Device registration
  - `devices/{deviceId}/heartbeat` - Keepalive
  - `devices/{deviceId}/status` - Status updates
  - `devices/{deviceId}/alerts` - Detection alerts
  - `devices/{deviceId}/snapshot` - Camera snapshots
  - `devices/{deviceId}/metrics` - Metrics data

- **Server → Device**:
  - `devices/{deviceId}/commands/metrics` - Metrics control
  - `devices/{deviceId}/commands/snapshot` - Snapshot request

### Schema References
```yaml
# Uses shared schemas
DeviceRegister:
  payload:
    $ref: '../schemas/register/v1.0.0/schema.json'

Snapshot:
  payload:
    $ref: '../schemas/snapshot/v1.0.0/schema.json'

MetricsData:
  payload:
    $ref: '../schemas/metrics-data/v1.0.0/schema.json'
```

## 3. WebSocket API - AsyncAPI Spec
**File**: [`shared/specs/asyncapi-websocket.yaml`](asyncapi-websocket.yaml)  
**Protocol**: WebSocket  
**Client**: Web browser (device monitoring UI)  
**Server**: ws WebSocket server

### Messages

**Client → Server**:
- `metrics_start` - Request metrics to start
- `metrics_stop` - Request metrics to stop

**Server → Client**:
- `connected` - Connection confirmation
- `snapshot` - Camera snapshot broadcast
- `metrics` - Metrics data broadcast

### Schema References
```yaml
# Uses shared schemas
SnapshotBroadcast:
  payload:
    $ref: '../schemas/snapshot/v1.0.0/schema.json'

MetricsBroadcast:
  payload:
    $ref: '../schemas/metrics-data/v1.0.0/schema.json'
```

## Schema Consistency

All three specifications reference the **same shared schemas** from `/shared/schemas/`:

| Message Type | REST API | MQTT API | WebSocket API |
|--------------|----------|----------|---------------|
| Device Info | ✅ `register/` | ✅ `register/` | ❌ N/A |
| Snapshot | ❌ N/A | ✅ `snapshot/` | ✅ `snapshot/` |
| Metrics Config | ✅ `metrics-config/` | ✅ `metrics-config/` | ❌ N/A |
| Metrics Data | ✅ `metrics-data/` | ✅ `metrics-data/` | ✅ `metrics-data/` |
| Alerts | ❌ N/A | ✅ `alerts/` | ❌ N/A |
| Heartbeat | ❌ N/A | ✅ `heartbeat/` | ❌ N/A |
| Status | ❌ N/A | ✅ `status/` | ❌ N/A |

## Runtime Validation

**Location**: [`web-platform/server/validation.ts`](../../web-platform/server/validation.ts)

Uses [AJV (Another JSON Validator)](https://ajv.js.org/) to validate messages at runtime:

```typescript
// WebSocket validation
validateClientMessage(message)   // Browser → Server
validateServerMessage(message)   // Server → Browser

// MQTT validation  
validateMQTTMessage(topic, payload)  // Device ↔ Server
```

**Benefits**:
- ✅ Security: Reject malformed messages
- ✅ Governance: Enforce contracts
- ✅ Debugging: Detailed error messages
- ✅ Type safety: Runtime type checking

## Message Flow Examples

### Example 1: Snapshot Request
```
Browser → REST POST /api/devices/{id}/snapshot
  ↓
Server → MQTT PUBLISH devices/{id}/commands/snapshot
  ↓
Device → MQTT PUBLISH devices/{id}/snapshot (with image)
  ↓
Server → WebSocket BROADCAST {type: 'snapshot', ...}
  ↓
Browser ← Receives snapshot via WebSocket
```

**Schemas Used**:
- MQTT command: `snapshot/v1.0.0/schema.json`
- MQTT response: `snapshot/v1.0.0/schema.json`
- WebSocket broadcast: `snapshot/v1.0.0/schema.json`

### Example 2: Metrics Lifecycle
```
Browser → WebSocket SEND {type: 'metrics_start'}
  ↓
Server → MQTT PUBLISH devices/{id}/commands/metrics {"enabled": true}
  ↓
Device → MQTT PUBLISH devices/{id}/metrics (every 60s)
  ↓
Server → WebSocket BROADCAST {type: 'metrics', data: {...}}
  ↓
Browser → WebSocket SEND {type: 'metrics_stop'}
  ↓
Server → MQTT PUBLISH devices/{id}/commands/metrics {"enabled": false}
```

**Schemas Used**:
- WebSocket: `command-simple/v1.0.0/schema.json`
- MQTT command: `metrics-config/v1.0.0/schema.json`
- MQTT data: `metrics-data/v1.0.0/schema.json`
- WebSocket broadcast: `metrics-data/v1.0.0/schema.json`

## Documentation Tools

Generate interactive docs from specs:

```bash
# OpenAPI (REST API)
npx @redocly/cli preview-docs shared/specs/openapi.yaml

# AsyncAPI (MQTT)
npx @asyncapi/generator shared/specs/asyncapi-mqtt.yaml @asyncapi/html-template

# AsyncAPI (WebSocket)
npx @asyncapi/generator shared/specs/asyncapi-websocket.yaml @asyncapi/html-template
```

## Keeping Specs in Sync

When adding new message types:

1. **Create schema** in `/shared/schemas/{name}/v1.0.0/schema.json`
2. **Reference in AsyncAPI/OpenAPI** specs
3. **Add validation** in `validation.ts`
4. **Update this README** with usage examples

See [`/shared/schemas/README.md`](../schemas/README.md) for schema guidelines.
