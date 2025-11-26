# Pipeline: Camera + Test prompts -> YOLO -> Clusterer -> FashionCLIP Vision + Text -> Compare -> Label
# Same as CLIP pipeline but using FashionCLIP for fashion-specific image-text matching

[camera(type="none"), test_input(prompts="chair,person driving a car,man wearing a white cowboy hat, blue shirt")]
-> detector(type="yolo_cpu", model="yolov8n.pt")
-> debug(types="detections", label="After YOLO")
-> detector(type="clusterer", max_clusters="4", merge_threshold="1.2", proximity_weight="1.7", size_weight="2.0", semantic_pair_weight="0.75")
-> debug(types="detections,audit", label="After Clusterer")
-> [
     fashion_clip_text_encoder(),
     fashion_clip_vision()
     :ordered_merge(order="0,1"):
   ]
-> clip_comparator(temperature="0.01") 
-> detection_labeler(min_probability="0.15", max_labels="3") 
-> viewer()
-> similarity_report(min_score="0.05") 
-> output()
