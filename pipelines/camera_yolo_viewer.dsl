# Pipeline: Camera -> YOLO Detection -> Viewer
# Captures image, runs YOLO object detection, displays results with bounding boxes

camera(type="none") -> detector(type="yolo_cpu", model="yolov8n.pt") -> viewer(timeout=5)
