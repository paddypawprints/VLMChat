# Simple Detection Pipeline Test
# Tests the basic detection and filtering architecture without CLIP
#
# Flow:
#   1. Collect text prompts from user (loop until exit code 1)
#   2. Capture image from camera  
#   3. Detect objects with YOLO
#   4. Filter to keep relevant categories
#   5. Display detections
#
# This tests the new detection_filter_task without requiring CLIP

{
  # Step 1: Collect text prompts and capture image
  [
    {input() -> break_on(code=1)},
    camera()
  ]
  
  # Step 2: Detect all objects
  -> detector(type="yolo_cpu")
  
  # Step 3: Filter to keep only relevant categories
  -> filter(categories="person,horse,bicycle,car,dog,cat,bird")
  
  # Step 4: Display results
  -> output(types="detections,text")
}
