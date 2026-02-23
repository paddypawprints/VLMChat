# VLMChat AI Coding Agent Instructions

## ⚠️ CRITICAL: Project Status

**ACTIVE PROJECTS** (under development):
- `camera-framework/` - Lightweight pipeline framework for edge AI vision
- `macos-device/` - macOS device implementation
- `shared/` - Contract definitions (JSON schemas, OpenAPI/AsyncAPI specs)
- `web-platform/` - React/Express management interface with real-time monitoring

**LEGACY PROJECT** (reference only - DO NOT MODIFY):
- `vlmchat/` - Original Python vision pipeline (kept for reference and learning)

## 🔒 SECURITY: Never Bypass Authentication or Validation

## NEVER NEVER NEVER generate code that masks or swallows exception without explicit instructions and then note the explicit approval in the code.

**SECURITY-CRITICAL PATTERNS - DO NOT MODIFY EVEN FOR TESTING**
- **OpenAPI Validation Middleware** (`web-platform/server/index.ts`) - Validates all REST API requests, must remain in place
- **MQTT Message Validation** (`web-platform/server/validation.ts`) - Runtime validation with AJV prevents malformed messages
- **Device Authentication** - PKI-based device auth with Ed25519 keys, never bypass or disable
- **Session Management** - Express session middleware with secure defaults, do not modify
- **Database Schema Validation** - Drizzle + Zod schemas enforce data integrity at runtime

These security layers are governance and security critical. Study them as examples of proper validation, but never remove or bypass them.

** THE BACK END SERVICES RUN VIA docker-compose - DO NOT RUN INDIVIDUALLY

The docker-compose commands must be run from the web-platform folder

** THIS IS THE docker-compose command line
Usage:  docker-compose [OPTIONS] COMMAND

Define and run multi-container applications with Docker

Options:
      --all-resources              Include all resources, even
                                   those not used by services
      --ansi string                Control when to print ANSI
                                   control characters
                                   ("never"|"always"|"auto")
                                   (default "auto")
      --compatibility              Run compose in backward
                                   compatibility mode
      --dry-run                    Execute command in dry run mode
      --env-file stringArray       Specify an alternate environment file
  -f, --file stringArray           Compose configuration files
      --parallel int               Control max parallelism, -1 for
                                   unlimited (default -1)
      --profile stringArray        Specify a profile to enable
      --progress string            Set type of progress output
                                   (auto, tty, plain, json, quiet)
      --project-directory string   Specify an alternate working
                                   directory
                                   (default: the path of the, first
                                   specified, Compose file)
  -p, --project-name string        Project name

Management Commands:
  bridge      Convert compose files into another model

Commands:
  attach      Attach local standard input, output, and error streams to a service's running container
  build       Build or rebuild services
  commit      Create a new image from a service container's changes
  config      Parse, resolve and render compose file in canonical format
  cp          Copy files/folders between a service container and the local filesystem
  create      Creates containers for a service
  down        Stop and remove containers, networks
  events      Receive real time events from containers
  exec        Execute a command in a running container
  export      Export a service container's filesystem as a tar archive
  images      List images used by the created containers
  kill        Force stop service containers
  logs        View output from containers
  ls          List running compose projects
  pause       Pause services
  port        Print the public port for a port binding
  ps          List containers
  publish     Publish compose application
  pull        Pull service images
  push        Push service images
  restart     Restart service containers
  rm          Removes stopped service containers
  run         Run a one-off command on a service
  scale       Scale services 
  start       Start services
  stats       Display a live stream of container(s) resource usage statistics
  stop        Stop services
  top         Display the running processes
  unpause     Unpause services
  up          Create and start containers
  version     Show the Docker Compose version information
  volumes     List volumes
  wait        Block until containers of all (or specified) services stop.
  watch       Watch build context for service and rebuild/refresh containers when files are updated

Run 'docker-compose COMMAND --help' for more information on a command.


## Project Overview

An edge AI platform for deploying vision-language models (VLMs) on devices like Raspberry Pi and NVIDIA Jetson. The system consists of:

1. **Camera Framework** (`camera-framework/`) - Lightweight pipeline framework for edge device AI vision
2. **Web Platform** (`web-platform/`) - React/Express management interface with real-time monitoring
3. **Contract-First Architecture** (`shared/`) - JSON schemas and OpenAPI/AsyncAPI specs as source of truth
4. **Legacy Pipeline** (`vlmchat/`) - Original vision processing framework (reference only)

## Critical Architecture Patterns

### Legacy vlmchat/ (Reference Only)

The `vlmchat/` directory contains the original pipeline implementation - **use only as reference, do not modify**.

**⚠️ IMPORTANT**: Read [ARCHITECTURE_STATUS.md](../ARCHITECTURE_STATUS.md) for details on the legacy architecture.

- **Modern**: `vlmchat.pipeline.*` - Fluent API with method chaining (reference pattern)
- **Deprecated**: `vlmchat.object_detector.*` - Old detection format (avoid in new code)

**Reference Detection Pattern** (from legacy vlmchat):
```python
from vlmchat.pipeline.detection import Detection, CocoCategory

detection = Detection(
    bbox=(x1, y1, x2, y2),          # NOT .box
    confidence=0.95,                 # NOT .conf
    category=CocoCategory.PERSON,    # NOT .object_category
    source_image=image_container
)
```

### Active camera-framework/

The `camera-framework/` is the current, lightweight pipeline implementation:

```python
from camera_framework import Runner, Context, BaseTask

class Camera(BaseTask):
    def process(self, ctx: Context):
        frame = capture_frame()
        ctx.append("frame", frame)

runner = Runner([Camera(), Display()])
runner.run_once()
```

## Contract-First Development

### Schema-Driven Code Generation

ALL data contracts live in `shared/` and generate client/server code:

```bash
# Regenerate TypeScript + Python from schemas
just generate

# Individual generations
just generate-ts      # → web-platform/shared/types/api.ts
just generate-python  # → device-sdk/edge_llm_client/models/
```

**Never manually edit generated code**. Update schemas in `shared/schemas/` or specs in `shared/specs/`, then regenerate.

### Schema Structure

- `shared/schemas/` - Flat JSON schemas (e.g., `register-v1.0.0.json`, `snapshot-v1.0.0.json`)
- `shared/specs/openapi.yaml` - REST API (web platform ↔ backend)
- `shared/specs/asyncapi.yaml` - MQTT topics (devices ↔ platform) and WebSocket (browser ↔ platform)

**Schema naming convention**: `{name}-v{major}.{minor}.{patch}.json` (e.g., `alerts-v1.0.0.json`)

**Example**: Adding a new message type requires updating the schema + spec, then running `just generate`.

## Development Workflows

### Build Commands (justfile)

```bash
just install          # Install all dependencies
just dev              # Start Docker Compose services
just dev-bg           # Start in background
just logs [service]   # View logs
just db-push          # Push database schema changes
just test             # Run all tests (TypeScript + Python)
just validate         # Validate OpenAPI + AsyncAPI specs
```

### Python Testing

Platform-aware test runner auto-detects hardware (Jetson, RPi, macOS):

```bash
# From project root
python tests/run_tests.py --smoke        # Fast unit tests (~seconds)
python tests/run_tests.py --integration  # Integration tests (~minutes)
python tests/run_tests.py --all          # Full suite

# Using pytest directly (vlmchat package - legacy reference only)
cd vlmchat && pytest
cd vlmchat && pytest -m unit             # Unit tests only
cd vlmchat && pytest -m integration      # Integration tests only
```

**Platform-specific tests auto-skip** on wrong hardware (e.g., `@pytest.mark.jetson` skipped on macOS).

### Web Platform Stack

- **Frontend**: React + TypeScript + shadcn/ui components + TailwindCSS
- **Backend**: Express.js + Drizzle ORM (PostgreSQL) + OpenAPI validation middleware
- **Real-time**: MQTT (Mosquitto) for device comms, WebSocket for browser updates
- **Future**: mediasoup for WebRTC video streaming (currently commented out)

**Database Migrations**: Use Drizzle Studio (`just db-studio`) or push changes (`just db-push`).

## Platform-Specific Considerations

### Edge Device Installation (Legacy vlmchat/)

Read [docs/INSTALL.md](../docs/INSTALL.md) for detailed platform setup of the legacy vlmchat package:

- **Base install**: Lightweight (`pip install .`)
- **Vision extras**: Install platform-specific PyTorch wheel FIRST, then `pip install .[vision]`
- **Jetson**: Requires JetPack-matched PyTorch wheel (follow NVIDIA docs)
- **Raspberry Pi**: Install `picamera2` via `pip install .[raspberrypi]`

**pyproject.toml** keeps heavy dependencies (torch, transformers, onnxruntime) as optional extras to support cross-platform builds.

### Camera Sources (Legacy vlmchat/)

Zero-copy camera implementation for Jetson (see [vlmchat/pipeline/sources/jetson_camera.py](../vlmchat/pipeline/sources/jetson_camera.py)):

- Uses GStreamer pipeline for Jetson: `nvarguscamerasrc` → `nvvidconv` → `appsink`
- Pre-allocated buffer pool prevents memory allocations during capture
- Ring buffer + pool interaction for bounded memory
- `cv2.retrieve()` writes directly into pre-allocated buffers

**Pattern**: Subclass `CameraSource` for platform-specific implementations.

## Configuration Files

- `config.json` - Device pipeline config (YOLO models, CLIP settings, camera params)
- `pyproject.toml` - Python package metadata with platform-specific extras
- `web-platform/package.json` - Node.js dependencies
- `justfile` - Build automation recipes

**Config locations are standardized**: Device configs in root, web configs in `web-platform/`.

## Key File Locations

- Pipeline tasks: `vlmchat/pipeline/tasks/` (legacy reference)
- Detection objects: `vlmchat/pipeline/detection.py` (legacy reference)
- Cache system: `vlmchat/pipeline/cache/` (legacy reference)
- Web routes: `web-platform/server/routes.ts`
- Database schema: `web-platform/shared/schema.ts` (Drizzle)
- MQTT client: `web-platform/server/mqtt.ts`

## Common Gotchas

1. **vlmchat/ is legacy** - Reference only, do not modify or use in new code
2. **Regenerate after schema changes** - Run `just generate` before testing
3. **Platform detection matters** - Tests/code may behave differently on Jetson vs macOS
4. **🔒 SECURITY: OpenAPI validation is security-critical** - Never modify/remove validator middleware in `web-platform/server/index.ts`
5. **🔒 SECURITY: Runtime validation protects all entry points** - AJV validators in `web-platform/server/validation.ts` prevent injection attacks
6. **🔒 SECURITY: Never bypass authentication for testing** - Use proper test fixtures with valid credentials instead
7. **Context data uses lists** - Pipeline context expects `list[T]`, not single values (legacy vlmchat pattern)
8. **Camera timing modes** - `STRICT` tasks fail on timeout, `LATENT_TOLERANT` skip frames adaptively (legacy vlmchat)

## Testing Patterns

```python
# Platform-aware test
@pytest.mark.jetson
def test_gstreamer_pipeline():
    # Only runs on Jetson hardware
    pass

# Integration test with proper markers
@pytest.mark.integration
def test_full_pipeline():
    # Runs with --integration flag
    pass
```

## Getting Help

- Architecture decisions: [ARCHITECTURE_STATUS.md](../ARCHITECTURE_STATUS.md)
- Pipeline design: [docs/PIPELINE.md](../docs/PIPELINE.md) (legacy vlmchat reference)
- Installation: [docs/INSTALL.md](../docs/INSTALL.md)
- API specs: [shared/specs/README.md](../shared/specs/README.md)
- Contributing: [docs/CONTRIBUTING.md](../docs/CONTRIBUTING.md)

## Quick Reference

| Task | Command | Notes |
|------|---------|-------|
| Start development | `just dev` | Starts web-platform + services |
| Run tests | `python tests/run_tests.py --smoke` | Platform-aware test runner |
| Regenerate code | `just generate` | Schema-driven TypeScript + Python |
| Validate specs | `just validate` | OpenAPI + AsyncAPI validation |
| Database changes | `just db-push` | Drizzle ORM migrations |
| View logs | `just logs` | Docker compose logs |
| Clean build artifacts | `just clean` | Remove generated files |
