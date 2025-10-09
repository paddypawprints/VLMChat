#!/usr/bin/env bash
set -euo pipefail

# scripts/install-jetson.sh
# Convenience installer for NVIDIA Jetson Orin Nano.
# This script expects you to supply a Jetson-compatible torch wheel URL and
# optionally an onnxruntime wheel URL. It will install them first, then the
# project using constraints-jetson.txt.
# Usage:
#   TORCH_WHEEL_URL="<url>" ONNX_WHEEL_URL="<url>" ./scripts/install-jetson.sh

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}
CONSTRAINTS_FILE="$(pwd)/constraints-jetson.txt"
TORCH_WHEEL_URL=${TORCH_WHEEL_URL:-}
ONNX_WHEEL_URL=${ONNX_WHEEL_URL:-}

if [ -z "$TORCH_WHEEL_URL" ]; then
  echo "ERROR: Please set TORCH_WHEEL_URL environment variable to your Jetson torch wheel URL."
  echo "Example: TORCH_WHEEL_URL=https://developer.nvidia.com/path/to/torch-wheel-for-jetson.whl $0"
  exit 1
fi

echo "Using python: $PYTHON"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Install the provided torch wheel first
pip install "$TORCH_WHEEL_URL"

# Optionally install a custom onnxruntime wheel
if [ -n "$ONNX_WHEEL_URL" ]; then
  pip install "$ONNX_WHEEL_URL"
fi

# Install base with constraints (will not touch torch)
pip install -c "$CONSTRAINTS_FILE" --no-deps .

# Install vision extras (transformers, onnxruntime if not already installed)
pip install -c "$CONSTRAINTS_FILE" .[vision]

# Optionally install jetson helpers
pip install -c "$CONSTRAINTS_FILE" .[jetson]

echo "Jetson installation complete. Activate your venv with: source $VENV_DIR/bin/activate"
