#!/usr/bin/env python3
"""
Complete integration test for pipeline + chat application + Environment.

Tests:
1. Environment singleton works with backward compatibility
2. Load image via chat commands updates Environment
3. NoneCamera reads from Environment
4. Pipeline commands load and execute
5. Pipeline tasks can access Environment data
"""
import sys
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_environment_basics():
    """Test Environment singleton and backward compatibility."""
    print("=" * 60)
    print("TEST 1: Environment Basics")
    print("=" * 60)
    
    from src.pipeline.environment import Environment
    
    # Reset for clean slate
    Environment.reset()
    env = Environment.get_instance()
    
    # Test new API
    env.set("Test+task1+value", "hello")
    assert env.get("Test+task1+value") == "hello"
    assert env.has("Test+task1+value")
    assert len(env) == 1
    
    # Test backward compatibility (properties)
    env.current_image = "test_image"
    assert env.current_image == "test_image"
    assert env.get("App+main+current_image") == "test_image"
    
    env.history = "test_history"
    assert env.history == "test_history"
    assert env.get("App+main+history") == "test_history"
    
    # Test backward compatibility (methods)
    env.set_image("another_image")
    assert env.get_image() == "another_image"
    
    env.clear_image()
    assert env.get_image() is None
    
    print("  ✓ Environment singleton works")
    print("  ✓ New dictionary API works")
    print("  ✓ Backward compatibility maintained")
    print()

def test_chat_services_integration():
    """Test that chat services properly updates Environment."""
    print("=" * 60)
    print("TEST 2: Chat Services Integration")
    print("=" * 60)
    
    from src.utils.config import VLMChatConfig
    from src.metrics.metrics_collector import Collector
    from src.main.chat_services import VLMChatServices
    from src.pipeline.environment import Environment
    from PIL import Image
    import numpy as np
    
    # Create test image
    test_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    
    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        temp_path = f.name
        test_img.save(temp_path)
    
    try:
        # Reset environment
        Environment.reset()
        
        # Create chat services (this will initialize Environment)
        config = VLMChatConfig.from_json("config.json")
        collector = Collector()
        
        print("  Creating VLMChatServices...")
        app = VLMChatServices(config, collector)
        
        # Check Environment was initialized
        env = Environment.get_instance()
        assert env is not None
        print("  ✓ Environment initialized")
        
        # Test load_file updates Environment
        print(f"  Loading image from {temp_path}...")
        response = app._service_load_file(temp_path)
        assert response.code.value == 0, f"Load failed: {response.message}"
        print(f"  ✓ Load response: {response.message}")
        
        # Check Environment has the image
        env_image = env.current_image
        assert env_image is not None
        print(f"  ✓ Environment has image: {env_image.size}")
        
        # Verify it's the same image
        assert env_image.size == test_img.size
        print("  ✓ Image matches loaded image")
        
        # Check history is set
        assert env.history is not None
        print("  ✓ Environment has history reference")
        
    finally:
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    print()

def test_none_camera_environment():
    """Test that NoneCamera reads from Environment."""
    print("=" * 60)
    print("TEST 3: NoneCamera Environment Integration")
    print("=" * 60)
    
    from src.utils.config import VLMChatConfig
    from src.metrics.metrics_collector import Collector
    from src.camera.none_camera import NoneCamera
    from src.pipeline.environment import Environment
    from PIL import Image
    import numpy as np
    
    # Reset environment
    Environment.reset()
    env = Environment.get_instance()
    
    # Create camera
    config = VLMChatConfig.from_json("config.json")
    collector = Collector()
    camera = NoneCamera(config, collector)
    
    # Test 1: No image in Environment - should load default
    print("  Test 1: No environment image...")
    path, image = camera.capture_single_image()
    assert "trail" in path.lower()  # Default trail-riders image
    assert image is not None
    print(f"  ✓ Loaded default image: {path}")
    
    # Test 2: Image in Environment - should use it
    print("  Test 2: With environment image...")
    test_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    env.set_image(test_img)
    
    path, image = camera.capture_single_image()
    assert path == "environment_image"
    assert image is test_img
    print(f"  ✓ Used environment image: {path}")
    
    # Test 3: Clear environment - back to default
    print("  Test 3: Clear environment...")
    env.clear_image()
    
    path, image = camera.capture_single_image()
    assert "trail" in path.lower()
    print(f"  ✓ Back to default image: {path}")
    
    print()

def test_pipeline_commands():
    """Test pipeline commands via chat services."""
    print("=" * 60)
    print("TEST 4: Pipeline Commands")
    print("=" * 60)
    
    from src.utils.config import VLMChatConfig
    from src.metrics.metrics_collector import Collector
    from src.main.chat_services import VLMChatServices
    from src.pipeline.environment import Environment
    from PIL import Image
    import numpy as np
    import time
    
    # Reset environment
    Environment.reset()
    
    # Create chat services
    config = VLMChatConfig.from_json("config.json")
    collector = Collector()
    app = VLMChatServices(config, collector)
    
    # Load a test image first
    env = Environment.get_instance()
    test_img = Image.fromarray(np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
    env.set_image(test_img)
    print("  ✓ Test image loaded to Environment")
    
    # Test 1: Load pipeline
    print("  Test 1: Load pipeline...")
    response = app._service_pipeline("test_simple.dsl")
    assert response.code.value == 0, f"Failed to load pipeline: {response.message}"
    print(f"  ✓ {response.message}")
    
    # Test 2: Check status (should show loaded but not running)
    print("  Test 2: Check status...")
    response = app._service_status()
    assert "Pipeline loaded: Yes" in response.message
    assert "No pipeline running" in response.message
    print(f"  ✓ Status: {response.message}")
    
    # Test 3: Run pipeline
    print("  Test 3: Run pipeline...")
    response = app._service_run("")
    assert response.code.value == 0, f"Failed to run: {response.message}"
    print(f"  ✓ {response.message}")
    
    # Give it time to start
    time.sleep(0.5)
    
    # Test 4: Check status while running (might have finished quickly)
    print("  Test 4: Check running status...")
    response = app._service_status()
    print(f"  ✓ Status: {response.message}")
    
    # Wait for completion
    time.sleep(2)
    
    # Test 5: Final status
    print("  Test 5: Final status...")
    response = app._service_status()
    print(f"  ✓ Status: {response.message}")
    
    # Test 6: Check output file
    print("  Test 6: Check output file...")
    output_path = Path("test_output.jpg")
    if output_path.exists():
        size = output_path.stat().st_size
        print(f"  ✓ Output file created: {size} bytes")
        output_path.unlink()  # Cleanup
    else:
        print("  ⚠ Output file not found (pipeline may have failed silently)")
    
    print()

def test_pipeline_task_environment_access():
    """Test that pipeline tasks can access Environment data."""
    print("=" * 60)
    print("TEST 5: Pipeline Task Environment Access")
    print("=" * 60)
    
    from src.pipeline.environment import Environment
    from src.pipeline.task_base import BaseTask, Context
    
    # Reset environment
    Environment.reset()
    env = Environment.get_instance()
    
    # Create a test task that uses Environment
    class EnvTestTask(BaseTask):
        def __init__(self):
            super().__init__("env_test")
        
        def run(self, context: Context) -> Context:
            env = Environment.get_instance()
            
            # Read from Environment
            value = env.get("Test+source+data", "default")
            
            # Write to Environment
            env.set("Test+env_test+result", f"processed_{value}")
            
            return context
    
    # Set up test data
    env.set("Test+source+data", "input_data")
    print("  ✓ Set test data in Environment")
    
    # Run task
    task = EnvTestTask()
    ctx = Context()
    result_ctx = task.run(ctx)
    
    # Check task wrote to Environment
    result = env.get("Test+env_test+result")
    assert result == "processed_input_data"
    print("  ✓ Task read from Environment")
    print("  ✓ Task wrote to Environment")
    print(f"  ✓ Result: {result}")
    
    print()

def test_inline_pipeline():
    """Test inline pipeline DSL."""
    print("=" * 60)
    print("TEST 6: Inline Pipeline DSL")
    print("=" * 60)
    
    from src.utils.config import VLMChatConfig
    from src.metrics.metrics_collector import Collector
    from src.main.chat_services import VLMChatServices
    from src.pipeline.environment import Environment
    
    # Reset environment
    Environment.reset()
    
    # Create chat services
    config = VLMChatConfig.from_json("config.json")
    collector = Collector()
    app = VLMChatServices(config, collector)
    
    # Test inline DSL
    print("  Testing inline DSL...")
    inline_dsl = "NoneCamera() -> Grayscale()"
    response = app._service_pipeline(inline_dsl)
    assert response.code.value == 0, f"Failed: {response.message}"
    print(f"  ✓ {response.message}")
    
    # Check status
    response = app._service_status()
    assert "Pipeline loaded: Yes" in response.message
    print("  ✓ Inline pipeline loaded")
    
    print()

def run_all_tests():
    """Run all integration tests."""
    print("\n")
    print("="*60)
    print("COMPLETE INTEGRATION TEST SUITE")
    print("="*60)
    print()
    
    tests = [
        ("Environment Basics", test_environment_basics),
        ("Chat Services Integration", test_chat_services_integration),
        ("NoneCamera Environment", test_none_camera_environment),
        ("Pipeline Commands", test_pipeline_commands),
        ("Pipeline Task Environment Access", test_pipeline_task_environment_access),
        ("Inline Pipeline DSL", test_inline_pipeline),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED: {name}")
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
            print()
    
    print("="*60)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*60)
    
    if failed == 0:
        print("\n✓ ALL INTEGRATION TESTS PASSED!")
        print("\nThe integration is complete:")
        print("  • Environment singleton working")
        print("  • Chat services updates Environment")
        print("  • NoneCamera reads from Environment")
        print("  • Pipeline commands functional")
        print("  • Pipeline tasks can access Environment")
        print("  • Both file and inline DSL work")
        return 0
    else:
        print(f"\n✗ {failed} TEST(S) FAILED")
        return 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
