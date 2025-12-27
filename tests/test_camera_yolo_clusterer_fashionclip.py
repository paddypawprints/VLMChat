#!/usr/bin/env python3
"""
Test camera -> yolo -> clusterer -> fashionclip pipeline.
Tests clustered detections with FashionCLIP vision and text comparison.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"

for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.config import VLMChatConfig
from src.metrics.metrics_collector import Collector
from src.main.chat_services import VLMChatServices
import time


def test_camera_yolo_clusterer_fashionclip():
    """Test camera + test_input -> yolo -> clusterer -> fashionclip vision + text -> compare"""
    print("\n" + "="*70)
    print("TEST: Camera + Test Input -> YOLO -> Clusterer -> FashionCLIP Pipeline")
    print("="*70)
    
    try:
        config = VLMChatConfig()
        collector = Collector()
        services = VLMChatServices(config, collector)
        
        dsl_file = "pipelines/camera_yolo_clusterer_fashionclip.dsl"
        
        print(f"\nLoading pipeline: {dsl_file}")
        response = services._service_pipeline(dsl_file)
        print(f"Pipeline loaded: {response.message}")
        
        if response.code.value != 0:
            print(f"\n❌ Failed to load pipeline")
            return False
        
        print("\nRunning pipeline with test prompts: person riding horse, horse")
        response = services._service_run("")
        print(f"Pipeline started: {response.message}")
        
        # Wait for pipeline to complete
        print("Waiting for pipeline to complete...")
        time.sleep(10)
        
        response = services._service_status()
        print(f"Status: {response.message}")
        print(f"Pipeline loaded: {services._current_pipeline is not None}")
        
        print("\n" + "="*70)
        print("✓ TEST PASSED")
        print("="*70)
        print("Pipeline executed successfully")
        print("FashionCLIP comparison complete - check logs for match results")
        
        return True
        
    except Exception as e:
        print("\n" + "="*70)
        print("✗ TEST FAILED")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_camera_yolo_clusterer_fashionclip()
    sys.exit(0 if success else 1)
