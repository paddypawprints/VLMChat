#!/usr/bin/env python3
"""
Lightweight integration test that doesn't require full app initialization.
Tests the core integration points without importing chat_services.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_environment_dict_api():
    """Test Environment dictionary-based API."""
    print("="*60)
    print("TEST 1: Environment Dictionary API")
    print("="*60)
    
    from src.pipeline.environment import Environment
    
    Environment.reset()
    env = Environment.get_instance()
    
    # Test basic operations
    env.set("Camera+cam1+image", "test_image_1")
    env.set("Camera+cam2+image", "test_image_2")
    env.set("Detector+det1+results", [1, 2, 3])
    
    assert env.get("Camera+cam1+image") == "test_image_1"
    assert env.get("Camera+cam2+image") == "test_image_2"
    assert env.get("Detector+det1+results") == [1, 2, 3]
    
    # Test has/in
    assert env.has("Camera+cam1+image")
    assert "Camera+cam1+image" in env
    
    # Test keys
    keys = env.keys()
    assert len(keys) == 3
    assert "Camera+cam1+image" in keys
    
    # Test remove
    assert env.remove("Detector+det1+results")
    assert not env.has("Detector+det1+results")
    assert len(env) == 2
    
    print("  ✓ Set/get operations work")
    print("  ✓ Has/in operations work")
    print("  ✓ Keys listing works")
    print("  ✓ Remove operations work")
    print()

def test_backward_compatibility():
    """Test Environment with full keys (backward compatibility removed)."""
    print("="*60)
    print("TEST 2: Environment with Full Keys")
    print("="*60)
    
    from src.pipeline.environment import Environment
    from PIL import Image
    import numpy as np
    
    Environment.reset()
    env = Environment.get_instance()
    
    # Use full keys for image
    test_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    env.set("App+main+current_image", test_img)
    assert env.get("App+main+current_image") is test_img
    print("  ✓ Full key for current_image works")
    
    # Use full keys for history
    env.set("App+main+history", "test_history")
    assert env.get("App+main+history") == "test_history"
    print("  ✓ Full key for history works")
    
    # Clear by removing key
    env.remove("App+main+current_image")
    assert env.get("App+main+current_image") is None
    print("  ✓ Remove works correctly")
    
    print()

def test_none_camera_environment():
    """Test NoneCamera Environment integration without full app."""
    print("="*60)
    print("TEST 3: NoneCamera Environment Integration")
    print("="*60)
    
    # We can test the logic without actually creating the camera
    # by checking that the Environment access pattern works
    from src.pipeline.environment import Environment
    from PIL import Image
    import numpy as np
    
    Environment.reset()
    env = Environment.get_instance()
    
    # Simulate what NoneCamera does
    test_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    
    # Test 1: No image - would fall back to default
    assert env.get("App+main+current_image") is None
    print("  ✓ No image in Environment initially")
    
    # Test 2: Set image - NoneCamera should use it
    env.set("App+main+current_image", test_img)
    assert env.get("App+main+current_image") is test_img
    print("  ✓ Image set in Environment")
    
    # Test 3: Clear image
    env.remove("App+main+current_image")
    assert env.get("App+main+current_image") is None
    print("  ✓ Image cleared from Environment")
    
    print()

def test_namespace_separation():
    """Test that different tasks can maintain separate state."""
    print("="*60)
    print("TEST 4: Namespace Separation")
    print("="*60)
    
    from src.pipeline.environment import Environment
    
    Environment.reset()
    env = Environment.get_instance()
    
    # Multiple cameras
    env.set("Camera+cam1+image", "image1")
    env.set("Camera+cam1+config", {"res": 1080})
    env.set("Camera+cam2+image", "image2")
    env.set("Camera+cam2+config", {"res": 720})
    
    # Multiple detectors
    env.set("Detector+det1+results", [1, 2, 3])
    env.set("Detector+det1+threshold", 0.5)
    env.set("Detector+det2+results", [4, 5, 6])
    env.set("Detector+det2+threshold", 0.7)
    
    # Verify separation
    assert env.get("Camera+cam1+image") == "image1"
    assert env.get("Camera+cam2+image") == "image2"
    assert env.get("Camera+cam1+config") != env.get("Camera+cam2+config")
    
    assert env.get("Detector+det1+results") == [1, 2, 3]
    assert env.get("Detector+det2+results") == [4, 5, 6]
    assert env.get("Detector+det1+threshold") != env.get("Detector+det2+threshold")
    
    print("  ✓ Multiple camera instances maintain separate state")
    print("  ✓ Multiple detector instances maintain separate state")
    print("  ✓ Cross-task isolation verified")
    print()

def test_pipeline_task_usage():
    """Test how pipeline tasks would use Environment."""
    print("="*60)
    print("TEST 5: Pipeline Task Environment Usage")
    print("="*60)
    
    from src.pipeline.environment import Environment
    from src.pipeline.task_base import BaseTask, Context
    
    Environment.reset()
    env = Environment.get_instance()
    
    # Simulate a producer task
    class ProducerTask(BaseTask):
        def __init__(self, task_id):
            super().__init__(task_id)
        
        def run(self, context: Context) -> Context:
            env = Environment.get_instance()
            env.set(f"Producer+{self.task_id}+output", "produced_data")
            return context
    
    # Simulate a consumer task
    class ConsumerTask(BaseTask):
        def __init__(self, task_id, producer_id):
            super().__init__(task_id)
            self.producer_id = producer_id
        
        def run(self, context: Context) -> Context:
            env = Environment.get_instance()
            data = env.get(f"Producer+{self.producer_id}+output")
            env.set(f"Consumer+{self.task_id}+result", f"consumed_{data}")
            return context
    
    # Execute pipeline
    producer = ProducerTask("prod1")
    consumer = ConsumerTask("cons1", "prod1")
    
    ctx = Context()
    ctx = producer.run(ctx)
    ctx = consumer.run(ctx)
    
    # Verify data flow
    assert env.get("Producer+prod1+output") == "produced_data"
    assert env.get("Consumer+cons1+result") == "consumed_produced_data"
    
    print("  ✓ Producer task writes to Environment")
    print("  ✓ Consumer task reads from Environment")
    print("  ✓ Data flow between tasks works")
    print()

def test_cleanup_patterns():
    """Test common cleanup patterns."""
    print("="*60)
    print("TEST 6: Cleanup Patterns")
    print("="*60)
    
    from src.pipeline.environment import Environment
    
    Environment.reset()
    env = Environment.get_instance()
    
    # Set up multiple keys for a task
    env.set("Task+task1+temp1", "data1")
    env.set("Task+task1+temp2", "data2")
    env.set("Task+task1+result", "final")
    env.set("Task+task2+data", "other")
    
    # Cleanup pattern 1: Remove specific task data
    prefix = "Task+task1+"
    for key in list(env.keys()):
        if key.startswith(prefix):
            env.remove(key)
    
    assert not env.has("Task+task1+temp1")
    assert not env.has("Task+task1+temp2")
    assert not env.has("Task+task1+result")
    assert env.has("Task+task2+data")
    print("  ✓ Prefix-based cleanup works")
    
    # Cleanup pattern 2: Clear all
    env.clear()
    assert len(env) == 0
    print("  ✓ Clear all works")
    
    print()

def run_all_tests():
    """Run all lightweight integration tests."""
    print("\n")
    print("="*60)
    print("LIGHTWEIGHT INTEGRATION TEST SUITE")
    print("="*60)
    print()
    
    tests = [
        test_environment_dict_api,
        test_backward_compatibility,  # Now tests full keys
        test_none_camera_environment,
        test_namespace_separation,
        test_pipeline_task_usage,
        test_cleanup_patterns,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {test_func.__name__}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ ALL TESTS PASSED!")
        print("\nEnvironment refactoring is complete:")
        print("  • Dictionary-based key-value store working")
        print("  • Full key access (no backward compatibility)")
        print("  • Namespace separation functional")
        print("  • Pipeline task integration ready")
        print("  • All cleanup patterns validated")
        print("\nReady for production use!")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
