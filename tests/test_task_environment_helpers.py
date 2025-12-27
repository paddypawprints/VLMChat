#!/usr/bin/env python3
"""
Test BaseTask environment helper methods.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.pipeline.task_base import BaseTask, Context
from src.pipeline.environment import Environment


class TestTask(BaseTask):
    """Simple test task to verify helper methods."""
    
    def run(self, context: Context) -> Context:
        return context


def test_env_set_and_get():
    """Test env_set and env_get helper methods."""
    print("Testing env_set and env_get...")
    
    # Reset environment
    Environment.reset()
    env = Environment.get_instance()
    
    # Create test task
    task = TestTask(task_id="test_task_1")
    
    # Use helper method to set
    task.env_set("output", "test_value")
    task.env_set("count", 42)
    
    # Verify using helper method
    assert task.env_get("output") == "test_value"
    assert task.env_get("count") == 42
    
    # Verify using direct Environment API
    assert env.get("TestTask", "test_task_1", "output") == "test_value"
    assert env.get("TestTask", "test_task_1", "count") == 42
    
    # Test default value
    assert task.env_get("nonexistent", "default") == "default"
    
    print("  ✓ env_set and env_get work correctly")


def test_env_has():
    """Test env_has helper method."""
    print("\nTesting env_has...")
    
    Environment.reset()
    task = TestTask(task_id="test_task_2")
    
    # Initially doesn't have key
    assert not task.env_has("data")
    
    # Set and check
    task.env_set("data", "value")
    assert task.env_has("data")
    
    # Other keys still don't exist
    assert not task.env_has("other_key")
    
    print("  ✓ env_has works correctly")


def test_env_remove():
    """Test env_remove helper method."""
    print("\nTesting env_remove...")
    
    Environment.reset()
    task = TestTask(task_id="test_task_3")
    
    # Set value
    task.env_set("temporary", "temp_value")
    assert task.env_has("temporary")
    
    # Remove it
    assert task.env_remove("temporary") == True
    assert not task.env_has("temporary")
    
    # Removing again returns False
    assert task.env_remove("temporary") == False
    
    print("  ✓ env_remove works correctly")


def test_task_isolation():
    """Test that different task instances have separate namespaces."""
    print("\nTesting task isolation...")
    
    Environment.reset()
    
    # Create multiple task instances
    task1 = TestTask(task_id="task_1")
    task2 = TestTask(task_id="task_2")
    
    # Set different values
    task1.env_set("result", "result_1")
    task2.env_set("result", "result_2")
    
    # Verify isolation
    assert task1.env_get("result") == "result_1"
    assert task2.env_get("result") == "result_2"
    
    # One task's data doesn't affect the other
    task1.env_set("unique_to_1", "data1")
    assert task1.env_has("unique_to_1")
    assert not task2.env_has("unique_to_1")
    
    print("  ✓ Task instances are properly isolated")


def test_class_name_in_key():
    """Test that class name is properly included in the key."""
    print("\nTesting class name inclusion...")
    
    Environment.reset()
    env = Environment.get_instance()
    
    task = TestTask(task_id="task_x")
    task.env_set("data", "value")
    
    # Check that the full key includes the class name
    keys = env.keys()
    expected_key = "TestTask+task_x+data"
    assert expected_key in keys
    
    # Verify we can access via full key too
    assert env.get("TestTask", "task_x", "data") == "value"
    
    print("  ✓ Class name is properly included in keys")


def test_multiple_tasks_same_class():
    """Test multiple instances of the same task class."""
    print("\nTesting multiple instances of same class...")
    
    Environment.reset()
    
    # Create multiple instances of same class
    tasks = [TestTask(task_id=f"task_{i}") for i in range(5)]
    
    # Set unique values
    for i, task in enumerate(tasks):
        task.env_set("index", i)
        task.env_set("name", f"task_{i}")
    
    # Verify each has its own data
    for i, task in enumerate(tasks):
        assert task.env_get("index") == i
        assert task.env_get("name") == f"task_{i}"
    
    print("  ✓ Multiple instances of same class work correctly")


def demo_usage():
    """Demonstrate usage patterns."""
    print("\n" + "="*60)
    print("USAGE DEMONSTRATIONS")
    print("="*60)
    
    Environment.reset()
    
    print("\n1. Basic task state storage:")
    print("   task = TestTask(task_id='processor')")
    print("   task.env_set('status', 'processing')")
    print("   status = task.env_get('status')")
    task = TestTask(task_id='processor')
    task.env_set('status', 'processing')
    print(f"   Retrieved: {task.env_get('status')}")
    
    print("\n2. Caching results between runs:")
    print("   if not task.env_has('cached_result'):")
    print("       result = expensive_computation()")
    print("       task.env_set('cached_result', result)")
    print("   return task.env_get('cached_result')")
    if not task.env_has('cached_result'):
        result = "computed_value"
        task.env_set('cached_result', result)
    print(f"   Retrieved from cache: {task.env_get('cached_result')}")
    
    print("\n3. Cleanup temporary data:")
    print("   task.env_set('temp_data', data)")
    print("   # ... use data ...")
    print("   task.env_remove('temp_data')")
    task.env_set('temp_data', 'temporary')
    print(f"   Before cleanup: has temp_data = {task.env_has('temp_data')}")
    task.env_remove('temp_data')
    print(f"   After cleanup: has temp_data = {task.env_has('temp_data')}")
    
    print("\n4. Default values for optional data:")
    print("   threshold = task.env_get('threshold', default=0.5)")
    threshold = task.env_get('threshold', default=0.5)
    print(f"   Retrieved threshold: {threshold}")


if __name__ == "__main__":
    print("="*60)
    print("BASETASK ENVIRONMENT HELPER TESTS")
    print("="*60)
    
    try:
        test_env_set_and_get()
        test_env_has()
        test_env_remove()
        test_task_isolation()
        test_class_name_in_key()
        test_multiple_tasks_same_class()
        demo_usage()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED!")
        print("="*60)
        print("\nBaseTask now provides:")
        print("  • env_set(key, value) - Store with auto taskType/taskId")
        print("  • env_get(key, default) - Retrieve with auto taskType/taskId")
        print("  • env_has(key) - Check existence with auto taskType/taskId")
        print("  • env_remove(key) - Remove with auto taskType/taskId")
        print("\nBenefits:")
        print("  • No need to specify class name or task_id manually")
        print("  • Consistent namespacing across all tasks")
        print("  • Cleaner, more readable task code")
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
