# Simple pipeline: Camera -> Image Viewer
# This pipeline captures an image from the camera and displays it
# Viewer handles images with or without detections

camera(type="none") -> viewer(timeout=3)
