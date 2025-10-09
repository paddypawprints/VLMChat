<!-- WARNING: Installation instructions are a work in progress and have not been fully tested across all target platforms. Use at your own risk and verify steps on your hardware. -->
INSTALLATION GUIDE for VLMChat

This document explains recommended installation flows for common targets:
- Local development (x86_64)
- Raspberry Pi / IMX500 (aarch64, Ubuntu + libcamera)
- NVIDIA Jetson Orin Nano (aarch64, Ubuntu, typically Python 3.10)

Overview
--------
VLMChat keeps heavyweight ML packages (PyTorch, ONNX runtime, Transformers) and
platform-specific camera libraries out of the base install. That makes the base
install lightweight and lets you install platform-matching binary wheels for
`torch` (and `onnxruntime`) before installing the project extras.

Basic / cross-platform (developer) install
-----------------------------------------
1. Create and activate a virtual environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

2. Install the base package (lightweight runtime deps):

```bash
pip install .
```

3. Install developer/test extras (optional):

```bash
pip install .[dev]
```

4. Install the ML/vision extras when you have a compatible `torch` or
   platform-specific wheel in place (see platform sections below):

```bash
pip install .[vision]
```

Platform-specific notes
-----------------------
Because both Raspberry Pi IMX500 and Jetson Orin Nano are aarch64 and run
Ubuntu, the project exposes separate extras so you can choose the right
components for each platform.

Raspberry Pi (aarch64, Ubuntu + libcamera)
-----------------------------------------
1. Ensure your OS has libcamera and the required kernel/drivers for your
   IMX500/IMX219 camera.

2. Create and activate a venv (see Basic section).

3. Install base + Raspberry Pi extras (this will attempt to install
   `picamera2` where appropriate):

```bash
pip install .[raspberrypi]
```

4. Install the Vision/ML extras after you have a compatible `torch` wheel
   for your Pi (many Pi users install platform-specific PyTorch wheels).

Note: The `transformers` package is available from PyPI for both Raspberry Pi
and Jetson; you can install it directly with `pip install transformers`.

```bash
# For Raspberry Pi you can typically use the regular PyPI CPU build of PyTorch
# (installing a CPU wheel from PyPI). Example:
pip install torch

# Install transformers from PyPI (safe on both Pi and Jetson)
pip install transformers

# Then install vision extras (onnxruntime will be skipped if already
# satisfied or you can install it separately first)
pip install .[vision]
```

Notes:
- If `pip` fails to find a compatible `torch` wheel for your Pi, check the
  official PyTorch documentation and community builds for ARM64.
- `picamera2` is platform-specific; installing it on non-Linux machines is not
  recommended.

Jetson Orin Nano (aarch64, Ubuntu — typically Python 3.10)
----------------------------------------------------------
NVIDIA Jetson devices usually require vendor-provided or JetPack-matched
PyTorch builds. The common pattern is:

1. Prepare JetPack and OS image according to NVIDIA's documentation.
2. Install a PyTorch wheel that matches your JetPack/CUDA combination. NVIDIA
   provides guidance and wheels; please follow their instructions for your
   JetPack/CUDA version.

Example (placeholder — follow official NVIDIA docs for exact commands):

```bash
# Acquire and install the Jetson-appropriate torch wheel first. Replace the
# URL below with the official wheel from the NVIDIA / Jetson developer site:
pip install https://developer.nvidia.com/path/to/torch-wheel-for-jetson.whl
```

3. Install the project without touching dependencies (so your torch wheel
   remains intact):

```bash
pip install --no-deps .
```

4. Install remaining vision extras (transformers, onnxruntime) after torch is
   present and verified:

```bash
# For Jetson, if you built a custom onnxruntime wheel and host it (for
# example on GitHub releases), install that wheel first. Replace the URL
# below with your release asset URL:
pip install https://github.com/<owner>/<repo>/releases/download/<tag>/onnxruntime-jetson-aarch64.whl

# Install transformers from PyPI (recommended on Jetson as well)
pip install transformers

# Then install vision extras (transformers will be skipped if already
# installed and onnxruntime will be skipped if the custom wheel is present)
pip install .[vision]
```

5. Optional Jetson helper packages:

```bash
pip install .[jetson]
```

Verification
------------
From an activated environment, verify core imports:

```bash
python -c "import sys; print(sys.version)"
python -c "import numpy, PIL, requests, pydantic, psutil; print('core ok')"
python -c "import torch; print('torch', torch.__version__)"  # if installed
python -c "import transformers; print('transformers OK')"   # if installed
python -c "import onnxruntime; print('onnxruntime OK')"     # if installed
python -c "import picamera2; print('picamera2 OK')"         # Pi only
```

Run the application
-------------------

```bash
# Create default config (optional)
python3 src/main.py --create-config

# Run the interactive CLI
python3 src/main.py

# Get ONNX model info
python3 src/main.py --onnx-info
```

Run the tests
-------------

Install dev extras then run pytest or the provided test runner:

```bash
pip install .[dev]
pytest
# or
python run_prompt_tests.py
```

Troubleshooting
---------------
- If `pip install .[vision]` fails due to `torch` incompatibility, install the
  correct `torch` wheel for your platform first and then re-run `pip install
  .[vision]` (or use `pip install --no-deps .` followed by selective installs).
- If `picamera2` import fails on a Pi, ensure libcamera and the camera kernel
  modules are installed and that your user has permission to access the
  camera device.
- For Jetson-specific issues (PyTorch / CUDA / JetPack mismatches), consult
  NVIDIA Jetson forums and the JetPack documentation for exact wheel versions.

Additional resources
--------------------
- Official PyTorch install guide: https://pytorch.org/get-started/locally/
- NVIDIA Jetson documentation and forums for JetPack and PyTorch on Jetson

If you'd like, I can:
- Create convenience scripts (`scripts/install-pi.sh` and `scripts/install-jetson.sh`) that walk the user through the recommended sequence (these would include placeholders for the correct torch wheel), or
- Add pinned versions for `transformers` / `onnxruntime` and provide example wheel URLs for the Jetson/ARM64 platforms.

Constraints and installer scripts
--------------------------------

I added constraint files and convenience install scripts to streamline reproducible installs:

- `constraints-pi.txt` — exact pins collected from the Raspberry Pi environment.
- `constraints-jetson.txt` — pins for Jetson installs; leaves torch/onnxruntime for manual installation (Jetson wheels are platform-specific).
- `scripts/install-pi.sh` — creates a venv and installs using `constraints-pi.txt` (installs base + vision + raspberrypi extras).
- `scripts/install-jetson.sh` — requires `TORCH_WHEEL_URL` environment variable (and optional `ONNX_WHEEL_URL`) to install Jetson-specific wheels first, then installs the project using `constraints-jetson.txt`.

Example usage (Pi):

```bash
./scripts/install-pi.sh
```

Example usage (Jetson):

```bash
TORCH_WHEEL_URL="https://developer.nvidia.com/path/to/torch-wheel-for-jetson.whl" \
ONNX_WHEEL_URL="https://github.com/<owner>/<repo>/releases/download/<tag>/onnxruntime-jetson-aarch64.whl" \
   ./scripts/install-jetson.sh
```

Note: Make the scripts executable in your shell if necessary:

```bash
chmod +x scripts/install-*.sh
```

Camera & venv helper for Raspberry Pi
-------------------------------------

If you had problems accessing the Pi camera from inside a virtual environment,
there is a helper script that automates the common fixes:

- `scripts/setup-pi-camera-venv.sh` will:
   - check for libcamera and picamera2 system packages and offer to install them (apt)
   - add your user to the `video` group so the camera device nodes are accessible
   - create a virtualenv with `--system-site-packages` so system-installed
      `picamera2` is visible inside the venv
   - optionally attempt to install `picamera2` into the venv if needed
   - run quick runtime checks (libcamera-hello and a small Picamera2 import/test)

Run the helper (it will prompt for sudo as needed):

```bash
./scripts/setup-pi-camera-venv.sh
```

If you prefer not to let the script install system packages, run it with:

```bash
./scripts/setup-pi-camera-venv.sh --no-install
```

After running, activate the created venv (default `.venv_camera`) and try a
small import test in Python:

```bash
source .venv_camera/bin/activate
python -c "from picamera2 import Picamera2; p=Picamera2(); print('OK', type(p))"
```

Jetson camera & venv helper
---------------------------

For Jetson devices where the camera pipeline is backed by NVIDIA Argus,
GStreamer and JetPack-installed OpenCV, you can use a similar helper that
creates a venv with system-site-packages and runs basic checks:

- `scripts/setup-jetson-camera-venv.sh` will:
   - create a venv with `--system-site-packages` so JetPack-installed
      `cv2`/GStreamer/Argus libraries are visible inside the venv
   - run quick checks for `cv2`, `gst-inspect-1.0` and the `nvarguscamerasrc`
      element
   - attempt a simple OpenCV VideoCapture test (may fail in headless systems)

Run it like this (optionally set PYTHON if you need a specific Python):

```bash
./scripts/setup-jetson-camera-venv.sh
# or
PYTHON=python3.10 ./scripts/setup-jetson-camera-venv.sh
```

After running, activate the created venv (default `.venv_jetson_camera`) and
try a quick import:

```bash
source .venv_jetson_camera/bin/activate
python -c "import cv2; print('cv2', cv2.__version__)"
```

CSI camera on Jetson (pinmux / jetson-io)
----------------------------------------

Many Jetson carrier boards require configuring the CPU pinmux so the CSI
camera connector is assigned to the camera interface. NVIDIA provides the
`jetson-io` utility under `/opt/nvidia` to set the pinmux for camera modules.

Typical steps to enable a CSI camera on Jetson:

1. Make sure your JetPack image is up-to-date and that you followed the
   carrier-board camera connection instructions for your board.

2. Run the Jetson IO utility (this will prompt you to select a configuration
   profile and write the pinmux configuration). The script lives under
   `/opt/nvidia` on JetPack installations; run it with sudo:

```bash
sudo /opt/nvidia/jetson-io/jetson-io.py
# or, depending on your image:
sudo /opt/nvidia/jetson-io/jetson-io
```

3. After applying the pinmux change the utility will typically request a
   reboot. Reboot the device so the new pin configuration takes effect:

```bash
sudo reboot
```

4. After reboot, verify that the camera device nodes and GStreamer elements
   are available (for example `/dev/video0` and the `nvarguscamerasrc`
   element):

```bash
# check device node
ls -l /dev/video*

# check GStreamer camera element
gst-inspect-1.0 nvarguscamerasrc
```

If these checks succeed, your Jetson board should be ready to run the
Argus/GStreamer camera pipeline. If anything fails, consult the Jetson Linux
Developer Guide and the Jetson-IO docs linked below.

Useful links
- Jetson IO utility and guides: https://developer.nvidia.com/embedded/jetson-io
- Jetson Linux Developer Guide (camera sections): https://docs.nvidia.com/jetson

*** End of INSTALL.md
