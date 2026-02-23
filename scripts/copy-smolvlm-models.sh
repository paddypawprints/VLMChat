#!/bin/bash
# Copy SmolVLM ONNX models from Jetson to local machine

set -e

# Configuration
JETSON_USER="${JETSON_USER:-patrick}"
JETSON_HOST="${JETSON_HOST:-jetson.local}"
JETSON_PATH="${JETSON_PATH:-~/onnx/SmolVLM2-256M-Instruct}"
LOCAL_PATH="${LOCAL_PATH:-~/onnx/SmolVLM2-256M-Instruct}"

echo "📦 Copying SmolVLM models from Jetson..."
echo "  Source: ${JETSON_USER}@${JETSON_HOST}:${JETSON_PATH}"
echo "  Dest:   ${LOCAL_PATH}"
echo ""

# Create local directory
mkdir -p "$(dirname "$LOCAL_PATH")"

# Copy with scp (recursive, preserve timestamps, show progress)
scp -r -p -v "${JETSON_USER}@${JETSON_HOST}:${JETSON_PATH}" "${LOCAL_PATH}"

echo ""
echo "✅ Models copied successfully!"
echo ""
echo "Contents:"
ls -lh "${LOCAL_PATH}"
