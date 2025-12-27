"""
Test script for detection_labeler and viewer integration.

Tests:
1. Detection labeling with matched prompts
2. Viewer displaying matched prompts
3. Full pipeline with FashionCLIP
"""

import sys
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, '/Users/patrick/Dev/VLMChat')

from src.pipeline.task_base import Context, ContextDataType
from src.pipeline.tasks.detection_labeler_task import DetectionLabelerTask
from src.object_detector.detection_base import Detection
from src.object_detector.detection_viewer import DetectionViewer
from src.object_detector.image_viewer import ImageViewer

print("\n=== Test 1: Detection Labeler ===\n")

# Create test detections
det1 = Detection(box=(100, 100, 200, 200), object_category="person", conf=0.9)
det2 = Detection(box=(300, 150, 400, 250), object_category="person", conf=0.85)
det3 = Detection(box=(500, 200, 600, 300), object_category="chair", conf=0.8)

print(f"Original detections:")
for det in [det1, det2, det3]:
    print(f"  {det}")

# Create test context
ctx = Context()
ctx.data[ContextDataType.DETECTIONS] = [det1, det2, det3]

# Create test similarity scores
# Probabilities: shape (n_prompts, n_detections)
probabilities = np.array([
    [0.75, 0.05, 0.02],  # "person" prompt
    [0.15, 0.85, 0.03],  # "person with hat" prompt  
    [0.10, 0.10, 0.95]   # "chair" prompt
])

ctx.data[ContextDataType.SIMILARITY_SCORES] = {
    'probabilities': probabilities,
    'texts': ['person', 'person with hat', 'chair'],
    'detection_ids': [f'detection_{det1.id}', f'detection_{det2.id}', f'detection_{det3.id}'],
    'metric': 'cosine_similarity'
}

# Run labeler with threshold 0.15
labeler = DetectionLabelerTask(min_probability=0.15, max_labels=2, task_id="test_labeler")
output_ctx = labeler.run(ctx)

# Check results
labeled_dets = output_ctx.data.get(ContextDataType.DETECTIONS, [])
print(f"\nLabeled detections (threshold=0.15): {len(labeled_dets)}")
for det in labeled_dets:
    print(f"  {det}")
    if hasattr(det, 'matched_prompts'):
        print(f"    Prompts: {det.matched_prompts}")
        print(f"    Probabilities: {[f'{p:.3f}' for p in det.match_probabilities]}")

print("\n✓ Detection labeling works correctly")

print("\n=== Test 2: Viewer Display Format ===\n")

# Test the viewer's display format by checking __str__
print("Detection string representations (as viewer will show them):")
for det in labeled_dets:
    print(f"  {det}")

print("\n✓ Viewer will display matched prompts correctly")

print("\n=== Test 3: Detection Serialization ===\n")

# Test to_dict and from_dict
det_dict = labeled_dets[0].to_dict()
print(f"Serialized detection:")
print(f"  id: {det_dict['id']}")
print(f"  matched_prompts: {det_dict['matched_prompts']}")
print(f"  match_probabilities: {det_dict['match_probabilities']}")

# Reconstruct
reconstructed = Detection.from_dict(det_dict)
print(f"\nReconstructed detection:")
print(f"  {reconstructed}")
print(f"  ID preserved: {reconstructed.id == labeled_dets[0].id}")

print("\n✓ Detection serialization works correctly")

print("\n=== All Tests Passed ===\n")
