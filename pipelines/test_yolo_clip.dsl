# Test: Direct YOLO to CLIP
[camera(type="none"), test_input(prompts="person riding horse,horse")]
-> detector(type="yolo_cpu", model="yolov8n.pt") 
-> [clip_text_encoder(), clip_vision():ordered_merge(order="0,1"):]
-> clip_comparator()
