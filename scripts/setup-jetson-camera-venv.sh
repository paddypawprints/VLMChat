#!/usr/bin/env bash
set -euo pipefail

# scripts/setup-jetson-camera-venv.sh
# Prepare a Python virtualenv on NVIDIA Jetson devices so it can access
# system-installed camera/GStreamer/OpenCV/Argus libraries (installed as
# site packages by JetPack). The script creates a venv with
# --system-site-packages and runs a few quick checks.
#
# Usage:
#   ./scripts/setup-jetson-camera-venv.sh
#   PYTHON=python3.10 ./scripts/setup-jetson-camera-venv.sh
#
# The script does not attempt to install JetPack components; it assumes
# GStreamer, Argus elements (nvarguscamerasrc), and cv2 are already present.

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv_jetson_camera}

echo "Using python: $PYTHON"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
  echo "Python executable '$PYTHON' not found in PATH." >&2
  exit 1
fi

if [ -d "$VENV_DIR" ]; then
  echo "Virtualenv $VENV_DIR already exists. Skipping creation."
else
  echo "Creating virtualenv at $VENV_DIR (with system-site-packages)."
  $PYTHON -m venv --system-site-packages "$VENV_DIR"
fi

echo "Activate the venv with: source $VENV_DIR/bin/activate"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Quick runtime checks
echo "\n-- Quick runtime checks --"

# Check OpenCV availability
python - <<'PY'
import sys
try:
    import cv2
    print('cv2 version:', getattr(cv2, '__version__', 'unknown'))
except Exception as e:
    print('cv2 import failed:', e)
    sys.exit(1)
PY

# Check gst-launch / gst-inspect and nvarguscamerasrc element
if command -v gst-inspect-1.0 >/dev/null 2>&1; then
  echo "gst-inspect-1.0 found: checking for nvarguscamerasrc element..."
  if gst-inspect-1.0 nvarguscamerasrc >/dev/null 2>&1; then
    echo "nvarguscamerasrc element is available in GStreamer."
  else
    echo "nvarguscamerasrc element not found. You may need JetPack/GStreamer camera plugins installed." >&2
  fi
else
  echo "gst-inspect-1.0 not found in PATH; GStreamer may not be installed or in PATH." >&2
fi

# Try to open a camera device via OpenCV (may fail in headless or permission-lacking env)
python - <<'PY'
import cv2, sys
cap = None
try:
    # Try common camera device indexes
    for idx in (0, 1):
        cap = cv2.VideoCapture(idx)
        if cap is not None and cap.isOpened():
            print(f'cv2 opened device index {idx}')
            cap.release()
            break
    else:
        print('cv2 could not open /dev/video device indices 0 or 1 (this may be expected in headless or permission-limited environments)')
except Exception as e:
    print('cv2.VideoCapture test raised:', e)
    if cap is not None:
        try: cap.release()
        except: pass
    sys.exit(1)
PY

echo "\nSetup complete. If system libraries are present (JetPack), the venv created with --system-site-packages should be able to import and use them."

echo "If you see permission errors when opening camera devices, ensure your user has access to the device nodes (usually group 'video') and that those device files exist (e.g., /dev/video0)."
