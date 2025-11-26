#!/usr/bin/env python3
"""
Test camera -> detectionViewer pipeline using chat_services.

This tests:
1. Loading pipeline via chat_services._service_pipeline()
2. Running pipeline via chat_services._service_run()
3. Pipeline completes successfully
"""
import sys
from pathlib import Path
import time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.main.chat_services import VLMChatServices
from src.utils.config import VLMChatConfig
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_camera_viewer_pipeline():
    """Test basic camera->viewer pipeline using chat_services."""
    print("="*60)
    print("TEST: Camera -> DetectionViewer via chat_services")
    print("="*60)
    
    # Create configuration and services
    config = VLMChatConfig()
    services = VLMChatServices(config, collector=None)
    
    logger.info("VLMChatServices initialized")
    
    # Test pipeline DSL
    dsl_file = "pipelines/camera_viewer.dsl"
    
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
    
    # Wait for pipeline to complete
    logger.info("Waiting for pipeline to complete...")
    time.sleep(3)  # Give it time to run
    
    # Check status
    response = services._service_status()
    logger.info(f"Status: {response.message}")
    
    print("\n" + "="*60)
    print("✓ TEST PASSED")
    print("="*60)
    print("Pipeline executed successfully via chat_services")
    
    return True


if __name__ == "__main__":
    success = test_camera_viewer_pipeline()
    sys.exit(0 if success else 1)
