"""
Test script for YOLO backends (TensorRT and Ultralytics).

This script tests both backend implementations to verify they work correctly
and produce consistent results.

Run with: 
    cd /path/to/VLMChat
    python src/models/Yolov8n/test_backends.py [options]

Or from src directory:
    cd /path/to/VLMChat/src
    python -m models.Yolov8n.test_backends [options]

Options:
    --backend {tensorrt,ultralytics,all}  Which backend to test (default: all)
    --config PATH                          Path to config file (default: config.json)
    --image PATH                          Path to test image (default: dinner.jpg)
    --confidence FLOAT                    Confidence threshold (default: 0.25)
    --iou FLOAT                          IoU threshold for NMS (default: 0.45)
    --verbose                            Enable verbose logging
"""

import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, Dict, Any, List

# Set up project root and add src to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Add src to path if not already there
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import cv2
from utils.config import VLMChatConfig
from models.Yolov8n.tensorrt_backend import TensorRTBackend
from models.Yolov8n.ultralytics_backend import UltralyticsBackend
from models.Yolov8n.runtime_base import YoloRuntimeBase
from object_detector.coco_categories import CocoCategory

logger = logging.getLogger(__name__)


def load_test_image(image_path: Path) -> np.ndarray:
    """
    Load a test image from the specified path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array in BGR format
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")
    
    # Load image in BGR format (OpenCV default)
    image = cv2.imread(str(image_path))  # type: ignore[attr-defined]
    
    if image is None:
        raise ValueError(f"Failed to load image from: {image_path}")
    
    logger.info(f"Loaded test image: {image.shape} from {image_path}")
    return image


def test_backend_availability(backend: YoloRuntimeBase, backend_name: str) -> bool:
    """
    Test if a backend is available.
    
    Args:
        backend: Backend instance to test
        backend_name: Name of the backend for logging
        
    Returns:
        True if backend is available, False otherwise
    """
    logger.info(f"Testing {backend_name} availability...")
    
    if not backend.is_available:
        logger.warning(f"✗ {backend_name} backend is not available")
        return False
    
    logger.info(f"✓ {backend_name} backend is available")
    return True


def test_prepare_image(
    backend: YoloRuntimeBase, 
    test_image: np.ndarray,
    backend_name: str
) -> Tuple[bool, np.ndarray, float, Dict[str, Any]]:
    """
    Test the prepare_image method.
    
    Args:
        backend: Backend instance to test
        test_image: Input test image
        backend_name: Name of the backend for logging
        
    Returns:
        Tuple of (success, blob, scale, metadata)
    """
    logger.info(f"Testing {backend_name} prepare_image()...")
    
    try:
        blob, scale, meta = backend.prepare_image(test_image)
        logger.info(f"✓ prepare_image() successful")
        logger.info(f"  - Blob shape: {blob.shape}")
        logger.info(f"  - Scale: {scale}")
        logger.info(f"  - Metadata: {meta}")
        return True, blob, scale, meta
    except Exception as e:
        logger.error(f"✗ prepare_image() failed: {e}", exc_info=True)
        return False, np.array([]), 0.0, {}


def test_infer(
    backend: YoloRuntimeBase,
    blob: np.ndarray,
    backend_name: str
) -> Tuple[bool, np.ndarray]:
    """
    Test the infer method.
    
    Args:
        backend: Backend instance to test
        blob: Preprocessed image blob
        backend_name: Name of the backend for logging
        
    Returns:
        Tuple of (success, raw_output)
    """
    logger.info(f"Testing {backend_name} infer()...")
    
    try:
        raw_output = backend.infer(blob)
        logger.info(f"✓ infer() successful")
        logger.info(f"  - Output shape: {raw_output.shape}")
        logger.info(f"  - Output dtype: {raw_output.dtype}")
        return True, raw_output
    except Exception as e:
        logger.error(f"✗ infer() failed: {e}", exc_info=True)
        return False, np.array([])


def test_decode_output(
    backend: YoloRuntimeBase,
    raw_output: np.ndarray,
    scale: float,
    meta: Dict[str, Any],
    backend_name: str,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Test the decode_output method.
    
    Args:
        backend: Backend instance to test
        raw_output: Raw model output
        scale: Scale factor from preprocessing
        meta: Metadata from preprocessing
        backend_name: Name of the backend for logging
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Tuple of (success, detections)
    """
    logger.info(f"Testing {backend_name} decode_output()...")
    
    try:
        detections = backend.decode_output(
            raw_output, 
            scale, 
            meta,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        logger.info(f"✓ decode_output() successful")
        logger.info(f"  - Number of detections: {len(detections)}")
        
        if detections:
            logger.info(f"  - Sample detections (top 3):")
            for i, det in enumerate(detections[:3]):
                # Get proper COCO category name if class_name is numeric
                class_name = det['class_name']
                if class_name.isdigit():
                    category = CocoCategory.from_id(int(class_name))
                    class_name = category.label if category else class_name
                
                logger.info(f"    {i+1}. {class_name} "
                          f"(conf: {det['confidence']:.3f}, "
                          f"bbox: [{det['bbox'][0]:.1f}, {det['bbox'][1]:.1f}, "
                          f"{det['bbox'][2]:.1f}, {det['bbox'][3]:.1f}])")
        
        return True, detections
    except Exception as e:
        logger.error(f"✗ decode_output() failed: {e}", exc_info=True)
        return False, []


def test_backend_full(
    backend_class,
    backend_name: str,
    config: VLMChatConfig,
    test_image: np.ndarray,
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Run full test suite on a specific backend.
    
    Args:
        backend_class: Backend class to instantiate
        backend_name: Name of the backend for logging
        config: Application configuration
        test_image: Test image to use
        confidence_threshold: Minimum confidence for detections
        iou_threshold: IoU threshold for NMS
        
    Returns:
        Tuple of (all_tests_passed, detections)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {backend_name} Backend")
    logger.info(f"{'='*60}")
    
    # Initialize backend
    try:
        backend = backend_class(config)
        logger.info(f"✓ Backend instantiated")
    except Exception as e:
        logger.error(f"✗ Failed to instantiate backend: {e}", exc_info=True)
        return False, []
    
    # Test availability
    if not test_backend_availability(backend, backend_name):
        return False, []
    
    # Test prepare_image
    success, blob, scale, meta = test_prepare_image(backend, test_image, backend_name)
    if not success:
        return False, []
    
    # Test infer
    success, raw_output = test_infer(backend, blob, backend_name)
    if not success:
        return False, []
    
    # Test decode_output
    success, detections = test_decode_output(
        backend, raw_output, scale, meta, backend_name,
        confidence_threshold, iou_threshold
    )
    if not success:
        return False, []
    
    logger.info(f"\n✓ All {backend_name} tests passed!")
    return True, detections


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test YOLO backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--backend',
        choices=['tensorrt', 'ultralytics', 'all'],
        default='all',
        help='Which backend to test (default: all)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        default=None,
        help='Path to config file (default: PROJECT_ROOT/config.json)'
    )
    
    parser.add_argument(
        '--image',
        type=Path,
        default=None,
        help='Path to test image (default: dinner.jpg in script directory)'
    )
    
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Confidence threshold (default: 0.25)'
    )
    
    parser.add_argument(
        '--iou',
        type=float,
        default=0.45,
        help='IoU threshold for NMS (default: 0.45)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def main():
    """Main test function."""
    args = parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("YOLO Backend Test Suite")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    # Determine config path
    if args.config:
        config_path = args.config
    else:
        config_path = PROJECT_ROOT / "config.json"
    
    # Load configuration
    try:
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}")
            logger.info("Using default configuration...")
            config = VLMChatConfig()
        else:
            config = VLMChatConfig.load_from_file(str(config_path))
            logger.info(f"✓ Configuration loaded from {config_path}")
    except Exception as e:
        logger.error(f"✗ Failed to load configuration: {e}", exc_info=True)
        return 1
    
    # Determine image path
    if args.image:
        image_path = args.image
    else:
        image_path = SCRIPT_DIR / "dinner.jpg"
    
    # Load test image
    try:
        test_image = load_test_image(image_path)
    except Exception as e:
        logger.error(f"✗ Failed to load test image: {e}", exc_info=True)
        return 1
    
    # Test results
    results: Dict[str, Tuple[bool, List[Dict[str, Any]]]] = {}
    
    # Determine which backends to test
    backends_to_test = []
    if args.backend in ('tensorrt', 'all'):
        backends_to_test.append(('tensorrt', TensorRTBackend))
    if args.backend in ('ultralytics', 'all'):
        backends_to_test.append(('ultralytics', UltralyticsBackend))
    
    # Run tests
    for backend_name, backend_class in backends_to_test:
        try:
            passed, detections = test_backend_full(
                backend_class,
                backend_name.title(),
                config,
                test_image,
                args.confidence,
                args.iou
            )
            results[backend_name] = (passed, detections)
        except Exception as e:
            logger.error(f"{backend_name.title()} backend test crashed: {e}", exc_info=True)
            results[backend_name] = (False, [])
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    
    for backend_name, (passed, detections) in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        det_count = f"({len(detections)} detections)" if passed else ""
        logger.info(f"{backend_name:20s}: {status:10s} {det_count}")
    
    # Return exit code
    if any(passed for passed, _ in results.values()):
        logger.info("\n✓ At least one backend is working!")
        return 0
    else:
        logger.error("\n✗ No backends are working!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
