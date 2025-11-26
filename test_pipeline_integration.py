#!/usr/bin/env python3
"""
Quick test script for pipeline integration with chat app.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config import VLMChatConfig
from src.metrics.metrics_collector import Collector
from src.main.chat_services import VLMChatServices

def test_pipeline_integration():
    """Test basic pipeline integration."""
    print("Loading config...")
    config = VLMChatConfig.from_json("config.json")
    
    print("Creating collector...")
    collector = Collector()
    
    print("Creating VLMChatServices...")
    app = VLMChatServices(config, collector)
    
    print("\n=== Testing Pipeline Commands ===\n")
    
    # Test 1: Load a test image
    print("1. Loading test image...")
    resp = app._service_load_file("src/tests/trail_riders.jpg")
    print(f"   Result: {resp.message}")
    assert resp.code.value == 0, f"Failed to load image: {resp.message}"
    
    # Test 2: Check environment
    from src.pipeline.environment import Environment
    env = Environment.get_instance()
    print(f"   Environment current_image: {env.current_image is not None}")
    assert env.current_image is not None, "Image not in environment"
    
    # Test 3: Load pipeline
    print("\n2. Loading pipeline...")
    resp = app._service_pipeline("test_simple.dsl")
    print(f"   Result: {resp.message}")
    assert resp.code.value == 0, f"Failed to load pipeline: {resp.message}"
    
    # Test 4: Check status
    print("\n3. Checking status...")
    resp = app._service_status()
    print(f"   Result: {resp.message}")
    assert "Pipeline loaded: Yes" in resp.message
    assert "No pipeline running" in resp.message
    
    # Test 5: Run pipeline
    print("\n4. Running pipeline...")
    resp = app._service_run("")
    print(f"   Result: {resp.message}")
    assert resp.code.value == 0, f"Failed to run pipeline: {resp.message}"
    
    # Give it a moment to start
    import time
    time.sleep(0.5)
    
    # Test 6: Check status while running
    print("\n5. Checking status while running...")
    resp = app._service_status()
    print(f"   Result: {resp.message}")
    
    # Test 7: Wait for completion
    print("\n6. Waiting for completion...")
    time.sleep(2)
    
    # Test 8: Check final status
    print("\n7. Checking final status...")
    resp = app._service_status()
    print(f"   Result: {resp.message}")
    
    # Test 9: Check output file
    print("\n8. Checking output file...")
    output_path = Path("test_output.jpg")
    if output_path.exists():
        print(f"   Output file exists: {output_path}")
        print(f"   File size: {output_path.stat().st_size} bytes")
        # Clean up
        output_path.unlink()
    else:
        print(f"   Warning: Output file not found: {output_path}")
    
    print("\n=== All Tests Passed! ===\n")

if __name__ == "__main__":
    test_pipeline_integration()
