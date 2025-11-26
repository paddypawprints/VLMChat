# Pipeline: Camera + Test prompts -> YOLO -> Clusterer -> CLIP Vision + Text -> Compare
# Captures image and provides test prompts, runs YOLO detection, 
# clusters detections, then uses CLIP to compare cluster embeddings with text prompts

[camera(type="none"), test_input(prompts="person riding horse,horse,chair,dog driving a car,man wearing a white cowboy hat")]
-> detector(type="yolo_cpu", model="yolov8n.pt")
-> debug(types="detections", label="After YOLO")
-> detector(type="clusterer", max_clusters="4", merge_threshold="1.2", proximity_weight="1.5", size_weight="1.5", semantic_pair_weight="0.5")
-> viewer()
-> debug(types="detections,audit", label="After Clusterer")
-> [
     clip_text_encoder(),
     clip_vision()
     :ordered_merge(order="0,1"):
   ]
-> clip_comparator(temperature="0.03") -> similarity_report(min_score="0.1") -> debug(types="text") -> output()
