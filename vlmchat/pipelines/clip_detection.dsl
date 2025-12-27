# CLIP Detection Pipeline
# Demonstrates semantic matching between text prompts and image detections
#
# Flow:
#   1. Collect text prompts from user (loop until empty input)
#   2. Capture image from camera
#   3. Detect objects with YOLO
#   4. Filter to keep relevant categories
#   5. Encode both TEXT and detections with CLIP (parallel)
#   6. Compare text embeddings vs detection embeddings
#   7. Display ranked similarity scores
#
# Usage:
#   python src/main.py
#   > /pipeline pipelines/clip_detection.dsl
#   > /run
#   
#   At the % prompt, enter text queries like:
#     % person riding a horse
#     % red shirt
#     % bicycle
#     % [press Enter on empty line to capture and process]
#
# Requirements:
#   - Camera connected
#   - YOLO model (yolov8n.pt)
#   - CLIP model configured

# Step 1: Collect text prompts (loop until empty input)
{input() -> :break_on(code=1):}

# Step 2: Capture image from camera
-> camera(type="none")

# Step 3: Detect all objects with YOLO
-> detector(type="yolo_cpu", model="yolov8n.pt")

# Step 4: Filter to keep only relevant categories
-> filter(categories="person,horse,bicycle,car,dog,cat,bird")

# Step 5: Parallel encoding - TEXT and detections
-> [
    clip_text_encoder(),
    clip_vision()
    :ordered_merge(order="0,1"):
  ]

# Step 6: Compare embeddings (expects nested structure from ordered_merge)
-> clip_comparator()

# Step 7: Display results
-> output(types="similarities", top_k="5")

