#!/usr/bin/env python3
"""
Test camera + input -> YOLO -> clusterer -> viewer pipeline.
"""

import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"

for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.config import VLMChatConfig
from src.metrics.metrics_collector import Collector
from src.main.chat_services import VLMChatServices


def test_camera_yolo_clusterer_viewer():
    """Test camera + test_input -> YOLO -> clusterer -> viewer pipeline."""
    print("\n" + "="*70)
    print("TEST: Camera + Test Input -> YOLO -> Clusterer -> Viewer Pipeline")
    print("="*70)
    
    try:
        config = VLMChatConfig()
        collector = Collector()
        services = VLMChatServices(config, collector)
        
        dsl_file = "pipelines/camera_yolo_clusterer_viewer.dsl"
        
        print(f"\nLoading pipeline: {dsl_file}")
        response = services._service_pipeline(dsl_file)
        print(f"Pipeline loaded: {response.message}")
        
        if response.code.value != 0:
            print(f"\n❌ Failed to load pipeline")
            return False
        
        print("\nRunning pipeline with test prompts: person riding horse, horse, cowboy")
        response = services._service_run("")
        print(f"Pipeline started: {response.message}")
        
        # Wait for pipeline to complete
        print("Waiting for pipeline to complete...")
        time.sleep(8)
        
        response = services._service_status()
        print(f"Status: {response.message}")
        
        print("\n" + "="*70)
        print("✓ TEST PASSED")
        print("="*70)
        print("Pipeline executed successfully")
        print("Check /tmp for viewer output PNG with clustered detections!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = test_camera_yolo_clusterer_viewer()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
