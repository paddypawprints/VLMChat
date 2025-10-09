# VLMChat

VLMChat is a small Visual-Language Model (VLM) chat application that integrates a SmolVLM wrapper, optional ONNX runtime acceleration, and camera support for edge devices (for example, IMX500 on Raspberry Pi-like platforms).

This repository contains the source, configuration helpers, utilities for camera & image handling, a lightweight chat application, and a comprehensive test suite.

Key points
- Multimodal chat using SmolVLM via Hugging Face Transformers and a small wrapper model
- Optional ONNX Runtime optimization for faster inference on supported platforms
- Picamera2-based camera integration when run on compatible Linux hardware
- Configurable via files, environment variables, and command-line switches

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