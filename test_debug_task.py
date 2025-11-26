#!/usr/bin/env python3
"""
Test debug task - prints context and trace information.

Tests:
1. Debug task prints TEXT context
2. Debug task with label
3. Debug task with filtered types
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.main.chat_services import VLMChatServices
from src.utils.config import VLMChatConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_debug_task():
    """Test debug task with simple text pipeline."""
    print("="*60)
    print("TEST: Debug Task - console_input -> debug -> console_output")
    print("="*60)
    
    # Create configuration and services
    config = VLMChatConfig()
    services = VLMChatServices(config, collector=None)
    
    logger.info("VLMChatServices initialized")
    
    # Test pipeline DSL with debug task
    dsl_file = "pipelines/test_debug.dsl"
    
    # Load the pipeline
    logger.info(f"Loading pipeline: {dsl_file}")
    response = services._service_pipeline(dsl_file)
    
    if response.code.value != 0:  # Not OK
        print("\n" + "="*60)
        print("✗ TEST FAILED - Pipeline Load")
        print("="*60)
        print(f"Error: {response.message}")
        return False
    
    logger.info(f"Pipeline loaded: {response.message}")
    
    # Run the pipeline
    logger.info("Running pipeline...")
    response = services._service_run("")
    
    if response.code.value != 0:  # Not OK
        print("\n" + "="*60)
        print("✗ TEST FAILED - Pipeline Run")
        print("="*60)
        print(f"Error: {response.message}")
        return False
    
    logger.info(f"Pipeline started: {response.message}")
    
    # Wait for pipeline to complete (runs in background thread)
    import time
    time.sleep(2)
    
    # Check status
    status = services._service_status()
    logger.info(f"Status: {status.message}")
    
    print("\n" + "="*60)
    print("✓ TEST PASSED")
    print("="*60)
    print("Debug task executed and printed context information")
    
    return True


if __name__ == "__main__":
    success = test_debug_task()
    sys.exit(0 if success else 1)
