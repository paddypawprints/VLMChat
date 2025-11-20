#!/usr/bin/env python3
"""
Test script for factory-based task creation with configure().

Tests that tasks can be created purely from DSL-style parameters
without needing dependency injection.
"""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

from pipeline.pipeline_factory import PipelineFactory
from pipeline.camera_task import CameraTask
from pipeline.detector_task import DetectorTask
from pipeline.console_input_task import ConsoleInputTask
from pipeline.task_base import Context, ContextDataType

print("=" * 70)
print("Factory Configure() Test Suite")
print("=" * 70)

# Create factory and register tasks
factory = PipelineFactory()
factory.register_task('camera', CameraTask)
factory.register_task('detector', DetectorTask)
factory.register_task('console_input', ConsoleInputTask)

# Test 1: CameraTask with configure()
print("\n[TEST 1] CameraTask.configure() with CameraFactory")
print("-" * 70)
try:
    camera = factory.create_task('camera', 'cam0', {
        'type': 'none',
        'device': '0',
        'resolution': '640x480'
    })
    print(f"✅ Camera task created: {camera.task_id}")
    print(f"   - Has camera instance: {camera.camera is not None}")
    print(f"   - Camera type: {type(camera.camera).__name__}")
    
    # Try running it
    context = Context()
    result = camera.run(context)
    print(f"   - Image captured: {ContextDataType.IMAGE in result.data}")
    if ContextDataType.IMAGE in result.data:
        img = result.data[ContextDataType.IMAGE]
        print(f"   - Image size: {img.size}")
except Exception as e:
    print(f"❌ Camera test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 2: DetectorTask with configure()
print("\n[TEST 2] DetectorTask.configure() - YOLO CPU")
print("-" * 70)
try:
    detector = factory.create_task('detector', 'yolo', {
        'type': 'yolo_cpu',
        'model': 'yolov8n.pt',
        'confidence': '0.25'
    })
    print(f"✅ Detector task created: {detector.task_id}")
    print(f"   - Has detector instance: {detector.detector is not None}")
    print(f"   - Detector type: {type(detector.detector).__name__}")
except Exception as e:
    print(f"❌ Detector test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: ConsoleInputTask with configure()
print("\n[TEST 3] ConsoleInputTask.configure()")
print("-" * 70)
try:
    console = factory.create_task('console_input', 'input1', {
        'prompt': 'Enter command: '
    })
    print(f"✅ Console input task created: {console.task_id}")
    print(f"   - Prompt text: '{console.prompt_text}'")
except Exception as e:
    print(f"❌ Console input test failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Pipeline with configured tasks
print("\n[TEST 4] Full pipeline with configure()-created tasks")
print("-" * 70)
try:
    from pipeline.pipeline_runner import PipelineRunner
    from pipeline.task_base import Connector
    
    # Create pipeline
    pipeline = Connector("config_test_pipeline")
    
    # Create tasks via factory with configure()
    cam = factory.create_task('camera', 'cam_configured', {
        'type': 'none',
        'device': '0'
    })
    det = factory.create_task('detector', 'det_configured', {
        'type': 'yolo_cpu',
        'model': 'yolov8n.pt',
        'confidence': '0.3'
    })
    
    # Build pipeline
    pipeline.add_task(cam)
    pipeline.add_task(det)
    pipeline.add_edge(cam, det)
    
    # Run pipeline
    runner = PipelineRunner(pipeline)
    context = Context()
    result = runner.run(context)
    
    print(f"✅ Pipeline executed successfully")
    print(f"   - Has image: {ContextDataType.IMAGE in result.data}")
    print(f"   - Has detections: {ContextDataType.DETECTIONS in result.data}")
    if ContextDataType.DETECTIONS in result.data:
        dets = result.data[ContextDataType.DETECTIONS]
        print(f"   - Detection count: {len(dets)}")
    
    runner.shutdown()
except Exception as e:
    print(f"❌ Pipeline test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("✅ All configure() tests completed!")
print("=" * 70)
