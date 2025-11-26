# Pipeline: Camera + Test prompts -> YOLO -> Clusterer -> Viewer
# Captures image and provides test prompts in parallel, runs YOLO detection, 
# clusters detections semantically based on prompts, displays results

[camera(type="none"), test_input(prompts="person riding horse,horse,cowboy")]
-> detector(type="yolo_cpu", model="yolov8n.pt") 
-> detector(type="clusterer", max_clusters="4") 
-> viewer(timeout=5)
