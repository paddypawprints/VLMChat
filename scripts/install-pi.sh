#!/usr/bin/env bash
set -euo pipefail

# scripts/install-pi.sh
# Convenience installer for Raspberry Pi (aarch64) using constraints-pi.txt
# Usage:
#   ./scripts/install-pi.sh
# or to use a custom python:
#   PYTHON=python3.11 ./scripts/install-pi.sh

PYTHON=${PYTHON:-python3}
VENV_DIR=${VENV_DIR:-.venv}
CONSTRAINTS_FILE="$(pwd)/constraints-pi.txt"

echo "Using python: $PYTHON"
$PYTHON -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel

# Install base with constraints
pip install -c "$CONSTRAINTS_FILE" .

# Install vision extras (this will install pinned transformers/onnxruntime/torch)
pip install -c "$CONSTRAINTS_FILE" .[vision]

# Install raspberrypi extras
pip install -c "$CONSTRAINTS_FILE" .[raspberrypi]

echo "Installation complete. Activate your venv with: source $VENV_DIR/bin/activate"
