# Test Pipeline: Detection Labeler + Viewer
# Tests YOLO -> Clusterer -> CLIP -> Comparator -> Labeler -> Viewer

[   
    #camera(type="none",image_path="https://img.freepik.com/free-photo/young-people-london-streets_23-2149377194.jpg"), 
    camera(type="none"), 
    test_input(prompts="skateboard,person wearing a brown jacket,horse,person riding a horse,person wearing a white hat")
]
-> detector(type="yolo_cpu", model="yolov8n.pt")
-> detector(type="clusterer", max_clusters="4", merge_threshold="1.2" , proximity_weight=".5", size_weight=".5", semantic_pair_weight="1.5")-> debug(types="audit")
#->detection_expander(expansion_factor=0.20)
-> [
     clip_text_encoder(),
     #fashion_clip_text_encoder(),
     clip_vision()
     #fashion_clip_vision()
     :ordered_merge(order="0,1"):
   ]
-> clip_comparator(temperature="0.03", probability_mode="softmax")
-> detection_labeler(min_probability="0.0", max_labels="2")
->viewer(show_children=false)
