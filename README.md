# VLMChat

VLMChat is a small Visual-Language Model (VLM) chat application that integrates a SmolVLM wrapper, optional ONNX runtime acceleration, and camera support for edge devices (for example, IMX500 on Raspberry Pi-like platforms).

This repository contains the source, configuration helpers, utilities for camera & image handling, a lightweight chat application, and a comprehensive test suite.

Key points
- Multimodal chat using SmolVLM via Hugging Face Transformers and a small wrapper model
- Optional ONNX Runtime optimization for faster inference on supported platforms
- Picamera2-based camera integration when run on compatible Linux hardware
- Configurable via files, environment variables, and command-line switches

Contributing

We welcome contributions from the community! This project follows a merit-based, open-governance model inspired by "The Apache Way." Our goal is to create a welcoming environment for collaboration.

Before you get started, please review the following documents:
- [CONTRIBUTING.md](./CONTRIBUTING.md) Our main guide for contributors. It details the development workflow, coding standards, and how to submit a pull request.
- [GOVERNANCE.md](./GOVERNANCE.md) Outlines our governance model, project roles, and the decision-making process.
These documents describe standards we expect all community members to follow.

All contributions require a signed [Contributor License Agreement](./CLA.md) (CLA), a standard practice for protecting both the contributor and the project. The process is automated via a bot on your first pull request.

<!-- WARNING: Installation instructions are a work in progress and have not been fully tested across all target platforms. Use at your own risk and verify steps on your hardware. -->
Installation (recommended)

This project now includes a `pyproject.toml` and can be installed with pip. Use the method that matches your environment.

1) Create a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2) Install the project (basic runtime):

```bash
pip install .
```

3) Install with development/test extras:

```bash
pip install .[dev]
```

4) Optional: ONNX GPU runtime on Linux

```bash
pip install .[onnx_gpu]
```


Platform notes and aarch64 specifics

- `picamera2` is only appropriate on Linux distributions with libcamera and compatible camera hardware (for example, Raspberry Pi or similar IMX500 boards). Because both Raspberry Pi (IMX500) and NVIDIA Jetson Orin Nano are aarch64 and run Ubuntu, we provide separate extras so you can pick the correct platform-specific components.

- `torch` and some model dependencies are large and often require platform-specific wheels. For Jetson Orin Nano (Python 3.10) follow NVIDIA's instructions to install a JetPack- and CUDA-matching PyTorch wheel (for example via NVIDIA PyIndex or custom wheel). After installing the platform-specific `torch` wheel, install the project extras.

Example install flows

Raspberry Pi (aarch64, Ubuntu with libcamera):

```bash
# create venv and activate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# install base + raspberrypi extras (picamera2 will be pulled, if available)
pip install .[raspberrypi]

# Install ML extras (do this after ensuring torch wheel is correct for your Pi)
pip install .[vision]
```

Jetson Orin Nano (aarch64, Ubuntu) — example pattern:

```bash
# create venv and activate
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install platform-specific torch wheel first (example placeholder):
# pip install <torch-wheel-for-jetson>

# Install the project without pulling a different torch:
pip install --no-deps .

# Then install the vision extras (transformers, onnxruntime) once torch is
# present and compatible; you can also selectively install onnxruntime-gpu
# if your Jetson supports it.
pip install .[vision]
```

Notes:
- The `jetson` extra provides Jetson helper packages (like `jetson-stats`) but
	does not install PyTorch for you. Follow NVIDIA's advice for PyTorch installation.
- If you intend to use ONNX with GPU support, prefer the `onnx_gpu` extra on
	Linux, but be sure your platform's CUDA/toolkit versions are compatible.

Quick start

```bash
# create a default config file
python3 src/main.py --create-config

# run the app (interactive CLI)
python3 src/main.py

# check ONNX model/onnxruntime info
python3 src/main.py --onnx-info
```

## Metrics

This repository includes a small, self-contained metrics system located at
`src/utils/metrics_collector.py`. It's intentionally lightweight and meant for
low-dependency in-process metrics collection (timeseries, bounded retention,
and simple instruments). Use it for local telemetry or to prototype metrics
before exporting to a dedicated backend.

Quick examples

- Register a time series (allowed attribute keys, optional bounds):

```py
from src.utils.metrics_collector import Collector

collector = Collector()
collector.register_timeseries("requests", registered_attribute_keys=["route"], max_count=256, ttl_seconds=600)
```

- Create a session and attach instruments:

```py
from src.utils.metrics_collector import Session, CounterInstrument

session = Session(collector)
counter = CounterInstrument("requests_counter", binding_attributes={"route": "/home"})
session.add_instrument(counter, "requests")
```

- Record datapoints:

```py
from src.utils.metrics_collector import ValueType

collector.add_datapoint("requests", ValueType.INT, 1, attributes={"route": "/home"})
# or convenience helper
collector.data_point("requests", {"route": "/home"}, 1)
```

- Time an operation with a context manager (records milliseconds):

```py
with collector.duration_timer("operation.duration", attributes={"route": "/home"}):
	do_work()
```

- Export a session to JSON:

```py
path = session.export_to_json("/tmp/metrics")
print("Wrote session export to", path)
```

For full usage, see `src/utils/metrics_collector.py` and the tests in `tests/`.

Running tests

The repository includes a test runner and pytest-based tests. After installing the `dev` extras:

```bash
pytest
# or use the provided test runner
python run_prompt_tests.py
```

Configuration

Configuration is managed via the pydantic-based config in `src/config.py`. You may supply values using:
- a config file (JSON or YAML)
- environment variables prefixed with `VLMCHAT_`
- command-line options

See `CONFIG.md` for detailed configuration guidance.

If you want platform-specific install instructions added to this README (for example, an ARM/CUDA matrix or Raspberry Pi notes), tell me which targets to include and I will add them.

# VLMChat
