# VLMChat Source Code

This directory contains the source code for the SmolVLM (Small Vision Language Model) chat application - an interactive multimodal chatbot that can analyze images and engage in conversations about them.

## Overview

VLMChat is built around the HuggingFace SmolVLM model with ONNX runtime optimization for efficient inference on edge devices like Raspberry Pi. The application supports real-time image capture via IMX500 camera, conversation history management, and flexible prompt formatting.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

- **main/**: Application entry point and main chat interface
- **models/**: SmolVLM model wrapper and configuration
- **prompt/**: Conversation history and prompt management
- **services/**: RAG service for metadata retrieval
- **utils/**: Utility functions for image processing and camera interface
- **tests/**: Unit tests for the application components

## Key Features
````markdown
# VLMChat Source Code

This directory contains the source code for the SmolVLM (Small Vision Language Model) chat application — an interactive multimodal chatbot that can analyze images and engage in conversations about them.

## Overview

VLMChat is built around the HuggingFace SmolVLM model with optional ONNX runtime optimization for efficient inference on edge devices. The application supports real-time image capture via IMX500 camera, conversation history management, and flexible prompt formatting.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

- **main/**: Application entry point and main chat interface
- **models/**: SmolVLM model wrapper and configuration
- **prompt/**: Conversation history and prompt management
- **services/**: RAG service for metadata retrieval
- **utils/**: Utility functions for image processing and camera interface
- **tests/**: Unit tests for the application components

## Key Features

- **Vision-Language Model**: Powered by SmolVLM for multimodal understanding
- **ONNX Optimization**: Optional ONNX runtime for faster inference
- **Camera Integration**: Direct integration with IMX500 camera on supported hardware
- **Conversation Management**: Configurable history limits and formatting
- **Image Loading**: Support for URLs, local files, and camera capture
- **Interactive Chat**: Command-line interface with slash commands

<!-- WARNING: Installation instructions are a work in progress and have not been fully tested across all target platforms. Use at your own risk and verify steps on your hardware. -->
## Installation

The project contains `pyproject.toml`. Follow the steps that match your environment.

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

3) Install development & test extras:

```bash
pip install .[dev]
```

4) Optional: install ONNX GPU runtime on Linux

```bash
pip install .[onnx_gpu]
```

### Platform notes

- `picamera2` (camera support) is only available on Linux systems with libcamera and compatible hardware. Because both Raspberry Pi IMX500 and NVIDIA Jetson Orin Nano are aarch64 Ubuntu variants, we added platform-specific extras in `pyproject.toml`:
	- `raspberrypi` extra will pull `picamera2` on aarch64 Linux
	- `jetson` extra contains Jetson-helper packages (does not install PyTorch)

- `torch` and model packages are large and often require platform-specific wheels. On Jetson Orin Nano you will typically use Python 3.10 and a JetPack-matched PyTorch wheel; install that first, then install the repository packages (see Quick Start examples below).

### Quick Start (platform examples)

Raspberry Pi (aarch64, Ubuntu + libcamera):

```bash
# venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install base + Pi extras (picamera2 will be pulled if available)
pip install .[raspberrypi]

# ML extras (after making sure torch wheel is correct for your Pi)
pip install .[vision]
```

Jetson Orin Nano (aarch64, Ubuntu; Python 3.10):

```bash
# venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# Install Jetson-specific PyTorch wheel per NVIDIA instructions
# pip install <torch-wheel-for-jetson>

# Install the project without overwriting torch
pip install --no-deps .

# Install vision extras (transformers, onnxruntime) after torch is present
pip install .[vision]
```

## Quick Start

```bash
# create a default config file
python3 src/main.py --create-config

# run the app (interactive CLI)
python3 src/main.py

# check ONNX model/onnxruntime info
python3 src/main.py --onnx-info
```

## Configuration

Configuration is handled by the pydantic-based system in `src/config.py` and can be provided via:
- config file (JSON/YAML)
- environment variables (prefix `VLMCHAT_`)
- command-line options

See the repository `CONFIG.md` for details.

## Testing

After installing the `dev` extras you can run tests with pytest or use the provided test runner.

```bash
pytest
# or
python run_prompt_tests.py
```

The test runner supports many modes (unit/integration/performance) and integrates with optional plugins such as `pytest-xdist`, `pytest-timeout`, and `pytest-benchmark` if installed.

## Notes for contributors

- Please file pull requests against `main` and include tests for new features.
- Use the `dev` extras for running tests and linters locally.

For details about the prompt module and test structure, see `src/tests/test_prompt/`.

## Metrics

This repository includes a lightweight metrics system at `src/utils/metrics_collector.py`.
It provides:

- TimeSeries registration (name + allowed attribute keys, optional bounds: max_count, ttl_seconds)
- Collector which accepts datapoints and enforces attribute validation and eviction
- Session which observes a Collector and holds Instruments
- Instrument base class and several concrete instruments (Counter, Count, Average,
  HistogramByAttribute, AverageDuration, Histogram, etc.)

See `src/utils/README.md` for quick examples on registering timeseries, creating
sessions/instruments, using the `DurationTimer` context manager, and exporting a
session to JSON.
