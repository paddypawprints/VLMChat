#!/usr/bin/env python3
"""
Test the refactored Environment with new API: env.set(taskType, taskId, key, value)
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.pipeline.environment import Environment

def test_basic_operations():
    """Test basic set/get/has/remove operations."""
    print("Testing basic operations...")
    env = Environment.get_instance()
    
    # Set some values
    env.set("Camera", "cam1", "current_image", "test_image_data")
    env.set("ObjectDetector", "det1", "results", [1, 2, 3])
    env.set("App", "main", "history", {"messages": []})
    
    # Get values
    assert env.get("Camera", "cam1", "current_image") == "test_image_data"
    assert env.get("ObjectDetector", "det1", "results") == [1, 2, 3]
    assert env.get("App", "main", "history") == {"messages": []}
    
    # Has
    assert env.has("Camera", "cam1", "current_image")
    assert not env.has("NonExistent", "key", "data")
    
    # Get with default
    assert env.get("NonExistent", "key", "data", "default") == "default"
    
    # Remove
    assert env.remove("ObjectDetector", "det1", "results") == True
    assert env.remove("NonExistent", "key", "data") == False
    assert not env.has("ObjectDetector", "det1", "results")
    
    print("  ✓ Basic operations work correctly")

def test_keys_and_len():
    """Test keys() and len() operations."""
    print("\nTesting keys and length...")
    env = Environment.get_instance()
    env.clear()
    
    env.set("Task1", "id1", "key1", "value1")
    env.set("Task2", "id2", "key2", "value2")
    env.set("Task3", "id3", "key3", "value3")
    
    keys = env.keys()
    assert len(keys) == 3
    assert len(env) == 3
    assert "Task1+id1+key1" in keys
    assert "Task2+id2+key2" in keys
    assert "Task3+id3+key3" in keys
    
    print("  ✓ Keys and length operations work correctly")

def test_namespace_separation():
    """Test that different task instances maintain separate state."""
    print("\nTesting namespace separation...")
    env = Environment.get_instance()
    env.clear()
    
    # Multiple camera instances
    env.set("Camera", "cam1", "current_image", "image1")
    env.set("Camera", "cam2", "current_image", "image2")
    
    # Multiple detector instances
    env.set("ObjectDetector", "det1", "threshold", 0.5)
    env.set("ObjectDetector", "det2", "threshold", 0.8)
    
    # Verify separation
    assert env.get("Camera", "cam1", "current_image") == "image1"
    assert env.get("Camera", "cam2", "current_image") == "image2"
    assert env.get("ObjectDetector", "det1", "threshold") == 0.5
    assert env.get("ObjectDetector", "det2", "threshold") == 0.8
    
    print("  ✓ Namespace separation works correctly")

def test_clear_and_reset():
    """Test clear and reset operations."""
    print("\nTesting clear and reset...")
    env = Environment.get_instance()
    env.clear()  # Start fresh
    
    env.set("Task1", "id1", "key1", "value1")
    env.set("Task2", "id2", "key2", "value2")
    assert len(env) == 2
    
    env.clear()
    assert len(env) == 0
    assert not env.has("Task1", "id1", "key1")
    
    # Reset the singleton
    Environment.reset()
    env = Environment.get_instance()
    assert len(env) == 0
    
    print("  ✓ Clear and reset work correctly")

def demo_usage_patterns():
    """Demonstrate common usage patterns."""
    print("\n" + "="*60)
    print("USAGE PATTERN DEMONSTRATIONS")
    print("="*60)
    
    env = Environment.get_instance()
    env.clear()
    
    print("\n1. Camera task storing current frame:")
    print("   env.set('Camera', 'cam1', 'current_image', image)")
    env.set("Camera", "cam1", "current_image", "<PIL.Image object>")
    
    print("\n2. Detector storing results for next task:")
    print("   env.set('ObjectDetector', 'det1', 'results', detections)")
    env.set("ObjectDetector", "det1", "results", [{"box": [0,0,100,100], "confidence": 0.95}])
    
    print("\n3. Display task retrieving detector results:")
    print("   results = env.get('ObjectDetector', 'det1', 'results')")
    results = env.get("ObjectDetector", "det1", "results")
    print(f"   Retrieved: {results}")
    
    print("\n4. Chat app storing history:")
    print("   env.set('App', 'main', 'history', history_manager)")
    env.set("App", "main", "history", {"messages": ["Hello", "World"]})
    
    print("\n5. Pipeline task accessing chat history:")
    print("   history = env.get('App', 'main', 'history')")
    history = env.get("App", "main", "history")
    print(f"   Retrieved: {history}")
    
    print("\n6. Listing all active keys:")
    print("   keys = env.keys()")
    print(f"   Keys: {env.keys()}")
    
    print("\n7. Checking if data exists before using:")
    print("   if env.has('Camera', 'cam1', 'current_image'):")
    if env.has("Camera", "cam1", "current_image"):
        print("      Image available!")

if __name__ == "__main__":
    print("="*60)
    print("ENVIRONMENT REFACTOR TESTS")
    print("="*60)
    
    try:
        test_basic_operations()
        test_keys_and_len()
        test_namespace_separation()
        test_clear_and_reset()
        demo_usage_patterns()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nThe Environment now supports:")
        print("  • Dictionary-based key-value store")
        print("  • Explicit parameters: set(taskType, taskId, key, value)")
        print("  • Namespaced keys: {taskType}+{taskId}+{key}")
        print("  • Multiple task instances with separate state")
        print("  • Clean, type-safe API")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
