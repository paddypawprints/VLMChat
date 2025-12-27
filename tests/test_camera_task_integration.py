#!/usr/bin/env python3
"""
Test CameraTask integration with Environment helper methods.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.pipeline.tasks.camera_task import CameraTask
from src.pipeline.task_base import Context, ContextDataType
from src.pipeline.environment import Environment
from src.camera.none_camera import NoneCamera
from src.utils.config import VLMChatConfig
from PIL import Image
import numpy as np


def test_camera_task_stores_in_environment():
    """Test that CameraTask stores image in environment using helper methods."""
    print("Testing CameraTask stores in environment...")
    
    Environment.reset()
    env = Environment.get_instance()
    
    # Create a simple test camera
    config = VLMChatConfig()
    camera = NoneCamera(config, None)
    
    # Create CameraTask
    task = CameraTask(camera=camera, task_id="cam1")
    
    # Run task
    context = Context()
    result = task.run(context)
    
    # Verify image is in context
    assert ContextDataType.IMAGE in result.data
    assert len(result.data[ContextDataType.IMAGE]) > 0
    
    # Verify image is in environment using direct API
    stored_image = env.get("CameraTask", "cam1", "current_image")
    assert stored_image is not None
    assert isinstance(stored_image, Image.Image)
    
    # Verify path is stored
    stored_path = env.get("CameraTask", "cam1", "image_path")
    assert stored_path is not None
    
    print(f"  ✓ CameraTask stored image: {stored_path}")
    print(f"  ✓ Image accessible via environment key: CameraTask+cam1+current_image")


def test_camera_task_helper_methods():
    """Test that CameraTask can use all environment helper methods."""
    print("\nTesting CameraTask helper methods...")
    
    Environment.reset()
    
    config = VLMChatConfig()
    camera = NoneCamera(config, None)
    task = CameraTask(camera=camera, task_id="cam2")
    
    # Test env_set and env_get
    task.env_set("test_data", "test_value")
    assert task.env_get("test_data") == "test_value"
    
    # Test env_has
    assert task.env_has("test_data")
    assert not task.env_has("nonexistent")
    
    # Test env_remove
    assert task.env_remove("test_data") == True
    assert not task.env_has("test_data")
    assert task.env_remove("test_data") == False
    
    print("  ✓ All helper methods work correctly")


def test_none_camera_reads_from_camera_task():
    """Test that NoneCamera can read images stored by CameraTask."""
    print("\nTesting NoneCamera reads from CameraTask...")
    
    Environment.reset()
    env = Environment.get_instance()
    config = VLMChatConfig()
    
    # Create a test image
    test_image = Image.new('RGB', (100, 100), color='red')
    
    # Store via CameraTask helper (simulating a pipeline CameraTask)
    camera_task = CameraTask(camera=None, task_id="pipeline_cam")
    camera_task.env_set("current_image", test_image)
    
    # Now NoneCamera should find it
    none_camera = NoneCamera(config, None)
    path, image = none_camera.capture_single_image()
    
    assert path == "environment_image"
    assert image == test_image
    
    print("  ✓ NoneCamera successfully reads from CameraTask environment")


def test_none_camera_reads_from_chat_app():
    """Test that NoneCamera prioritizes chat app images."""
    print("\nTesting NoneCamera prioritizes chat app...")
    
    Environment.reset()
    env = Environment.get_instance()
    config = VLMChatConfig()
    
    # Create two test images
    pipeline_image = Image.new('RGB', (100, 100), color='blue')
    chat_image = Image.new('RGB', (100, 100), color='green')
    
    # Store pipeline image
    camera_task = CameraTask(camera=None, task_id="pipeline_cam")
    camera_task.env_set("current_image", pipeline_image)
    
    # Store chat app image (should have priority)
    env.set("App", "main", "current_image", chat_image)
    
    # NoneCamera should return chat app image (green)
    none_camera = NoneCamera(config, None)
    path, image = none_camera.capture_single_image()
    
    assert path == "environment_image"
    assert image == chat_image
    
    print("  ✓ NoneCamera correctly prioritizes chat app images")


def test_multiple_camera_tasks():
    """Test multiple CameraTask instances maintain separate state."""
    print("\nTesting multiple CameraTask instances...")
    
    Environment.reset()
    config = VLMChatConfig()
    
    # Create multiple camera tasks
    cam1 = CameraTask(camera=NoneCamera(config, None), task_id="cam1")
    cam2 = CameraTask(camera=NoneCamera(config, None), task_id="cam2")
    cam3 = CameraTask(camera=NoneCamera(config, None), task_id="cam3")
    
    # Run each task
    context = Context()
    cam1.run(context)
    cam2.run(context)
    cam3.run(context)
    
    # Each should have separate environment data
    assert cam1.env_has("current_image")
    assert cam2.env_has("current_image")
    assert cam3.env_has("current_image")
    
    # Add custom data to each
    cam1.env_set("resolution", "1920x1080")
    cam2.env_set("resolution", "640x480")
    cam3.env_set("resolution", "320x240")
    
    # Verify isolation
    assert cam1.env_get("resolution") == "1920x1080"
    assert cam2.env_get("resolution") == "640x480"
    assert cam3.env_get("resolution") == "320x240"
    
    print("  ✓ Multiple CameraTask instances properly isolated")


def test_environment_keys_format():
    """Test that environment keys follow correct format."""
    print("\nTesting environment key format...")
    
    Environment.reset()
    env = Environment.get_instance()
    config = VLMChatConfig()
    
    # Create and run camera task
    camera = NoneCamera(config, None)
    task = CameraTask(camera=camera, task_id="test_cam")
    task.run(Context())
    
    # Check keys follow format: CameraTask+task_id+key
    keys = env.keys()
    expected_image_key = "CameraTask+test_cam+current_image"
    expected_path_key = "CameraTask+test_cam+image_path"
    
    assert expected_image_key in keys
    assert expected_path_key in keys
    
    print(f"  ✓ Keys follow correct format:")
    print(f"    - {expected_image_key}")
    print(f"    - {expected_path_key}")


def demo_integration():
    """Demonstrate the integration."""
    print("\n" + "="*60)
    print("INTEGRATION DEMONSTRATION")
    print("="*60)
    
    Environment.reset()
    env = Environment.get_instance()
    config = VLMChatConfig()
    
    print("\n1. CameraTask captures and stores image:")
    print("   camera = NoneCamera(config, collector)")
    print("   task = CameraTask(camera, 'main_cam')")
    print("   context = task.run(Context())")
    
    camera = NoneCamera(config, None)
    task = CameraTask(camera=camera, task_id="main_cam")
    context = task.run(Context())
    
    print("\n2. Image stored in both context and environment:")
    print(f"   - Context: {len(context.data[ContextDataType.IMAGE])} image(s)")
    print(f"   - Environment key: CameraTask+main_cam+current_image")
    
    print("\n3. Other tasks can access via helper methods:")
    print("   # From another CameraTask instance")
    print("   other_task = CameraTask(..., 'other_cam')")
    print("   image = other_task.env_get('current_image')  # Gets 'other_cam' image")
    
    print("\n4. Or access cross-task via direct Environment API:")
    print("   env = Environment.get_instance()")
    print("   main_image = env.get('CameraTask', 'main_cam', 'current_image')")
    main_image = env.get("CameraTask", "main_cam", "current_image")
    print(f"   Retrieved: {type(main_image).__name__} @ {id(main_image)}")
    
    print("\n5. NoneCamera finds images from any source:")
    print("   Priority order:")
    print("   1. App+main+current_image (chat /load commands)")
    print("   2. CameraTask+*+current_image (pipeline tasks)")
    print("   3. trail-riders.jpg (fallback default)")
    
    # Demonstrate priority
    none_cam = NoneCamera(config, None)
    path1, img1 = none_cam.capture_single_image()
    print(f"   Current source: {path1}")
    
    # Add chat app image
    chat_image = Image.new('RGB', (100, 100), color='yellow')
    env.set("App", "main", "current_image", chat_image)
    path2, img2 = none_cam.capture_single_image()
    print(f"   After chat /load: {path2}")


if __name__ == "__main__":
    print("="*60)
    print("CAMERATASK ENVIRONMENT INTEGRATION TESTS")
    print("="*60)
    
    try:
        test_camera_task_stores_in_environment()
        test_camera_task_helper_methods()
        test_none_camera_reads_from_camera_task()
        test_none_camera_reads_from_chat_app()
        test_multiple_camera_tasks()
        test_environment_keys_format()
        demo_integration()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nIntegration complete:")
        print("  • CameraTask uses env_set/env_get/env_has/env_remove helpers")
        print("  • Images stored as: CameraTask+{task_id}+current_image")
        print("  • NoneCamera reads from Environment (chat app or pipeline)")
        print("  • Multiple camera instances properly isolated")
        print("  • Seamless integration between chat and pipeline systems")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
