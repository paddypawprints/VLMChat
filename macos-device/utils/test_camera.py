#!/usr/bin/env python3
"""Simple script to trigger macOS camera permission dialog."""

import cv2
import sys
import time

print("Attempting to access camera...")
print("This will trigger macOS permission dialog if not already granted.")
print()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Failed to open camera")
    print("Please grant camera permissions in System Settings > Privacy & Security > Camera")
    sys.exit(1)

print("✅ Camera opened successfully!")

# Wait for camera to initialize
time.sleep(1)

# Read one frame to confirm it works
ret, frame = cap.read()
if ret:
    print(f"✅ Frame captured: {frame.shape}")
else:
    print("❌ Failed to capture frame")
    cap.release()
    sys.exit(1)

cap.release()
print()
print("Camera permissions are working! You can now run the main application.")
