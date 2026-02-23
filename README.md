# VLMChat

An edge AI platform for deploying vision-language models (VLMs) on devices like Raspberry Pi and NVIDIA Jetson, with a web-based management interface.

## Project Structure

```
VLMChat/
├── camera-framework/    # Lightweight pipeline framework for edge AI vision
├── macos-device/        # macOS development/testing device implementation
├── shared/              # Contract definitions (source of truth)
│   ├── schemas/         # Versioned JSON schemas
│   └── specs/           # OpenAPI & AsyncAPI specifications
├── web-platform/        # React/Express management interface
├── device-sdk/          # Auto-generated Python SDK (from shared/schemas)
└── vlmchat/             # ⚠️ LEGACY — reference only, do not modify
```

## Active Components

### `camera-framework/` — Pipeline Framework
Lightweight, task-based pipeline framework for edge AI vision. Tasks share data through a `Context` object; a `Runner` executes them in order.

```bash
pip install -e camera-framework/
```

See [camera-framework/README.md](camera-framework/README.md).

### `macos-device/` — macOS Device
Full vision pipeline for macOS (development/testing). Runs YOLO detection, attribute enrichment, clustering, tracking, and MQTT publishing.

```bash
pip install -e camera-framework/
pip install -e macos-device/

# Run from project root (config files must be present)
python -m macos_device              # Standalone (no MQTT)
python -m macos_device --mqtt       # Full pipeline with MQTT
python -m macos_device --diagram    # Generate pipeline diagram
```

Requires `camera_framework_config.yaml` and `macos_device_config.yaml` in the working directory. See [macos-device/README.md](macos-device/README.md).

### `web-platform/` — Management Interface
React + Express web app with real-time device monitoring, MQTT bridging, and PostgreSQL persistence. Runs via Docker Compose.

```bash
cd web-platform
docker-compose up          # Start all services
docker-compose up -d       # Background
docker-compose logs -f     # View logs
```

Or using `just` from the project root:
```bash
just dev        # Start services
just dev-bg     # Start in background
just logs       # View logs
just stop       # Stop services
```

### `shared/` — Contracts
All API and message schemas live here. Never edit generated code — update schemas and regenerate:

```bash
just generate       # Regenerate TypeScript types + Python SDK
just validate       # Validate OpenAPI + AsyncAPI specs
```

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Docker + Docker Compose
- [`just`](https://github.com/casey/just): `brew install just`

### Install

```bash
# Python packages
pip install -e camera-framework/
pip install -e macos-device/

# Web platform dependencies
cd web-platform && npm install
```

### Run the device pipeline

```bash
# From project root
python -m macos_device --mqtt
```

### Run the web platform

```bash
just dev
```

## Configuration

| File | Purpose |
|------|---------|
| `camera_framework_config.yaml` | Pipeline settings (workers, memory thresholds, camera/image source) |
| `macos_device_config.yaml` | Device settings (YOLO model path, attribute model, MQTT broker, color filter) |
| `web-platform/docker-compose.yml` | Service definitions (Node, PostgreSQL, Mosquitto) |

## Development

```bash
just generate           # Regenerate code from schemas
just diagram            # Regenerate pipeline topology diagram
just db-push            # Push database schema changes
just db-studio          # Open Drizzle Studio
just test               # Run all tests
just validate           # Validate API specs
just clean              # Remove generated files
```

## Testing

```bash
# Python (camera-framework)
cd camera-framework && pytest

# Python (macos-device)
cd macos-device && pytest

# TypeScript (web-platform)
cd web-platform && npm test
```

## Legacy

`vlmchat/` contains the original vision pipeline implementation. It is kept for reference only — **do not modify or use in new code**. The active pipeline is `camera-framework/` + `macos-device/`.

## Contributing

See [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md) and [docs/GOVERNANCE.md](docs/GOVERNANCE.md). All contributions require a signed [CLA](docs/CLA.md).
