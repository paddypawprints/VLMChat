"""
Test script for the unified YoloObjectDetector.

This tests the detector with both TensorRT and Ultralytics backends,
verifying integration with the ObjectDetector interface.

Run with: python src/object_detector/test_yolo_object_detector.py
"""

import sys
import logging
from pathlib import Path

# Set up paths
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import cv2
from PIL import Image as PILImage

from utils.config import VLMChatConfig
from object_detector.yolo_object_detector import YoloObjectDetector

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_test_image() -> PILImage.Image:
    """Load the test image."""
    # Try to find dinner.jpg in models/Yolov8n directory
    image_path = SRC_DIR / "models" / "Yolov8n" / "dinner.jpg"
    
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")
    
    # Load with PIL (RGB format)
    image = PILImage.open(image_path)
    logger.info(f"Loaded test image: {image.size} from {image_path}")
    return image


def test_detector(runtime: str, config: VLMChatConfig, test_image: PILImage.Image):
    """Test the YoloObjectDetector with a specific runtime."""
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing YoloObjectDetector with {runtime} backend")
    logger.info(f"{'='*60}")
    
    try:
        # Create detector
        detector = YoloObjectDetector(
            config=config,
            runtime=runtime,
            confidence_threshold=0.25,
            iou_threshold=0.45
        )
        logger.info("✓ Detector instantiated")
        
        # Start detector
        detector.start()
        logger.info("✓ Detector started")
        
        # Check readiness
        if not detector.readiness():
            logger.error(f"✗ Detector not ready")
            return False
        logger.info("✓ Detector is ready")
        
        # Perform detection
        detections = detector.detect(test_image)  # type: ignore[arg-type]
        logger.info(f"✓ Detection completed: {len(detections)} objects found")
        
        # Display results
        if detections:
            logger.info("  Sample detections (top 5):")
            for i, det in enumerate(detections[:5]):
                logger.info(f"    {i+1}. {det.object_category} "
                          f"(conf: {det.conf:.3f}, "
                          f"box: {det.box})")
        else:
            logger.warning("  No detections found")
        
        # Stop detector
        detector.stop()
        logger.info("✓ Detector stopped")
        
        logger.info(f"\n✓ All tests passed for {runtime} backend!")
        return True
        
    except Exception as e:
        logger.error(f"✗ Test failed: {e}", exc_info=True)
        return False


def main():
    """Main test function."""
    logger.info("YoloObjectDetector Test Suite")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    # Load configuration
    try:
        config_path = PROJECT_ROOT / "config.json"
        if config_path.exists():
            config = VLMChatConfig.load_from_file(str(config_path))
            logger.info(f"✓ Configuration loaded from {config_path}")
        else:
            logger.warning("Config not found, using defaults")
            config = VLMChatConfig()
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}")
        return 1
    
    # Load test image
    try:
        test_image = load_test_image()
    except Exception as e:
        logger.error(f"✗ Failed to load test image: {e}")
        return 1
    
    # Test both backends
    results = {}
    
    for runtime in ['tensorrt', 'ultralytics']:
        try:
            results[runtime] = test_detector(runtime, config, test_image)
        except Exception as e:
            logger.error(f"{runtime} test crashed: {e}", exc_info=True)
            results[runtime] = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    
    for runtime, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        logger.info(f"{runtime:20s}: {status}")
    
    if any(results.values()):
        logger.info("\n✓ At least one backend is working!")
        return 0
    else:
        logger.error("\n✗ No backends are working!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
