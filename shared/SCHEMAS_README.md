# Shared Message Schemas

## Overview

This directory contains **protocol-agnostic JSON schemas** used across all communication layers:
- **MQTT** (Device ↔ Server)
- **WebSocket** (Browser ↔ Server)  
- **REST API** (future HTTP endpoints)

## Design Principles

1. **Single Source of Truth**: Each message type has ONE schema, regardless of transport protocol
2. **Reusability**: The same `snapshot` schema is used whether sent via MQTT or WebSocket
3. **Versioning**: All schemas are versioned (`v1.0.0`) for backward compatibility
4. **Validation**: Runtime validation enforces security and governance

## Schema Organization

```
shared/schemas/
├── alerts/v1.0.0/schema.json           # Detection alerts
├── command-simple/v1.0.0/schema.json   # Simple command (type only)
├── connection-ack/v1.0.0/schema.json   # WebSocket connection ACK
├── heartbeat/v1.0.0/schema.json        # Device heartbeat
├── metrics-config/v1.0.0/schema.json   # Metrics configuration
├── metrics-data/v1.0.0/schema.json     # Metrics data payload
├── register/v1.0.0/schema.json         # Device registration
├── snapshot/v1.0.0/schema.json         # Camera snapshot
├── status/v1.0.0/schema.json           # Device status
├── watchlist/v1.0.0/schema.json        # Watchlist configuration
└── webrtc-signaling/v1.0.0/schema.json # WebRTC signaling
```

## Usage

### AsyncAPI Specifications

Schemas are referenced in AsyncAPI specs using relative paths:

**MQTT Spec** (`shared/specs/asyncapi-mqtt.yaml`):
```yaml
messages:
  Snapshot:
    payload:
      $ref: '../schemas/snapshot/v1.0.0/schema.json'
```

**WebSocket Spec** (`shared/specs/asyncapi-websocket.yaml`):
```yaml
messages:
  SnapshotBroadcast:
    payload:
      $ref: '../schemas/snapshot/v1.0.0/schema.json'
```

### Runtime Validation

Server-side validation uses AJV (JSON Schema validator):

```typescript
import { validateClientMessage, validateServerMessage } from './validation';

// Validate WebSocket message from browser
const validation = validateClientMessage(message);
if (!validation.valid) {
  console.error('Invalid message:', validation.errors);
  return;
}

// Validate before broadcasting to clients
const validation = validateServerMessage(broadcastMessage);
if (!validation.valid) {
  console.error('Invalid broadcast:', validation.errors);
  return;
}
```

## Schema Structure

All schemas follow JSON Schema Draft 07:

```json
{
  "$schema": "http://json-schema.org/draft-07/schema#",
  "$id": "https://vlmchat.dev/schemas/{schema-name}/v1.0.0/schema.json",
  "title": "Human Readable Title",
  "description": "Detailed description",
  "type": "object",
  "required": ["field1", "field2"],
  "properties": {
    "field1": { "type": "string" },
    "field2": { "type": "number" }
  },
  "additionalProperties": false
}
```

## Adding New Schemas

1. **Create directory**: `shared/schemas/{schema-name}/v1.0.0/`
2. **Add schema.json**: Follow structure above
3. **Reference in AsyncAPI**: Update relevant spec files
4. **Add validation**: Update `web-platform/server/validation.ts`
5. **Test**: Validate with sample messages

## Schema Reuse Examples

### Snapshot (Used in MQTT + WebSocket)

**MQTT**: Device publishes to `devices/{id}/snapshot`
- Schema: `snapshot/v1.0.0/schema.json`
- Contains: `device_id`, `timestamp`, `image`, `format`

**WebSocket**: Server broadcasts to browser
- **Same schema**: `snapshot/v1.0.0/schema.json`
- Server wraps with `{type: 'snapshot', ...schema}`

### Metrics Data (Used in MQTT + WebSocket)

**MQTT**: Device publishes to `devices/{id}/metrics`
- Schema: `metrics-data/v1.0.0/schema.json`
- Contains: `session`, `timestamp`, `instruments`

**WebSocket**: Server broadcasts to browser
- **Same schema**: `metrics-data/v1.0.0/schema.json`
- Server wraps with `{type: 'metrics', data: {...schema}}`

## Benefits

✅ **Consistency**: Same validation rules across all protocols
✅ **Maintainability**: Update schema once, affects all usages
✅ **Documentation**: Single schema = single source of documentation
✅ **Type Safety**: Generate TypeScript types from schemas
✅ **Security**: Runtime validation prevents malformed messages
✅ **Governance**: Centralized contract enforcement

## Related Documentation

- [AsyncAPI MQTT Spec](../specs/asyncapi-mqtt.yaml)
- [AsyncAPI WebSocket Spec](../specs/asyncapi-websocket.yaml)
- [Validation Implementation](../../web-platform/server/validation.ts)
