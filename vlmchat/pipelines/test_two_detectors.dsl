# Test two detectors in sequence
[camera(type="none"), test_input(prompts="person riding horse")]
-> detector(type="yolo_cpu", model="yolov8n.pt") 
-> detector(type="clusterer", max_clusters="4")
-> viewer()
