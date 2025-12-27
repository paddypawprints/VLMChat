# Shared Contracts

This directory contains the **single source of truth** for all data contracts and API specifications across the VLMChat platform.

## Structure

```
shared/
├── schemas/          # JSON Schema definitions (versioned)
│   ├── alerts/
│   ├── device-register/
│   ├── metrics-config/
│   ├── metrics-data/
│   ├── watchlist/
│   └── webrtc-signaling/
└── specs/           # API specifications (to be added)
    ├── openapi.yaml      # REST API specification
    └── asyncapi.yaml     # MQTT/WebSocket specification
```

## Schemas

All schemas are versioned using semantic versioning (e.g., `v1.0.0/schema.json`).

### Current Schemas

- **alerts**: Detection alerts and system events from devices
- **device-register**: Device registration and configuration
- **metrics-config**: Metrics collection configuration
- **metrics-data**: Instrumentation and performance metrics
- **watchlist**: Object detection watchlist items
- **webrtc-signaling**: WebRTC signaling for real-time video

## Usage

### TypeScript (web-platform)

Import schemas via the `@shared` alias:
```typescript
import alertSchema from '@shared/schemas/alerts/v1.0.0/schema.json';
```

### Python (device-sdk)

Generated Pydantic models from these schemas:
```python
from edge_llm_client.models import Alert, MetricsData
```

## Code Generation

Run `just generate` from the project root to generate:
- TypeScript types for the web platform
- Python Pydantic models for device SDK
- MQTT client code from AsyncAPI spec
- REST API clients from OpenAPI spec

## Schema Guidelines

1. **Versioning**: Always create new versions for breaking changes
2. **Validation**: Test schemas with both TypeScript and Python validators
3. **Documentation**: Include clear descriptions in schema properties
4. **References**: Use `$ref` to share common definitions
5. **Compatibility**: Maintain backwards compatibility within major versions
