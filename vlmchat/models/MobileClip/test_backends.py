"""
Test script for CLIP backends (OpenCLIP and TensorRT image and text encoders).

This script tests backend implementations to verify they work correctly
and produce consistent results.

Run with: 
    cd /path/to/VLMChat
    python src/models/MobileClip/test_backends.py [options]

Or from src directory:
    cd /path/to/VLMChat/src
    python -m models.MobileClip.test_backends [options]

Options:
    --backend {image,text,all}           Which backend to test (default: all)
    --implementation {openclip,tensorrt} Which implementation to use (default: openclip)
    --config PATH                        Path to config file (default: config.json)
    --image PATH                         Path to test image (default: trail-riders.jpg)
    --prompts "text1" "text2" ...       Text prompts to test (default: ["a horse", "a person", "a dog"])
    --verbose                            Enable verbose logging
"""

import sys
import logging
import argparse
import numpy as np
from pathlib import Path
from typing import Tuple, List, Union

# Set up project root and add src to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Add src to path if not already there
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import torch
from PIL import Image
from utils.config import VLMChatConfig
from models.MobileClip.openclip_image_backend import OpenClipImageBackend
from models.MobileClip.openclip_text_backend import OpenClipTextBackend
from models.MobileClip.runtime_base import ClipImageRuntimeBase, ClipTextRuntimeBase

# Try importing TensorRT backends
try:
    from models.MobileClip.tensorrt_image_backend import TensorRTImageBackend
    from models.MobileClip.tensorrt_text_backend import TensorRTTextBackend
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False

logger = logging.getLogger(__name__)


def load_test_image(image_path: Path) -> Image.Image:
    """
    Load a test image from the specified path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        PIL Image
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")
    
    image = Image.open(image_path)
    logger.info(f"Loaded test image: {image.size} from {image_path}")
    return image


def test_backend_availability(
    backend: ClipImageRuntimeBase | ClipTextRuntimeBase, 
    backend_name: str
) -> bool:
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


def prepare_image_for_backend(
    test_image: Image.Image, 
    backend: ClipImageRuntimeBase
) -> Union[Image.Image, np.ndarray, torch.Tensor]:
    """
    Prepare image in the format preferred by the backend.
    
    For GPU tensor formats, uses the backend's preprocessor to ensure
    proper resizing and normalization.
    
    Args:
        test_image: PIL Image to convert
        backend: Backend that declares its native_image_format
        
    Returns:
        Image in backend's preferred format (PIL, numpy, or preprocessed torch tensor)
    """
    native_format = getattr(backend, 'native_image_format', 'pil')
    
    logger.info(f"  - Backend prefers format: {native_format}")
    
    if native_format == 'pil':
        logger.info(f"  - Providing PIL image")
        return test_image
    
    elif native_format == 'numpy':
        logger.info(f"  - Converting PIL → NumPy")
        return np.array(test_image)
    
    elif native_format == 'torch_cpu':
        logger.info(f"  - Converting PIL → Torch CPU tensor (preprocessed)")
        # Use backend's preprocessor if available
        if hasattr(backend, 'preprocess') and backend.preprocess is not None:  # type: ignore[attr-defined]
            return backend.preprocess(test_image.convert("RGB"))  # type: ignore[attr-defined]
        else:
            # Fallback: basic conversion
            np_image = np.array(test_image).astype(np.float32) / 255.0
            if np_image.ndim == 3:
                np_image = np_image.transpose(2, 0, 1)  # HWC → CHW
            return torch.from_numpy(np_image)
    
    elif native_format == 'torch_gpu':
        logger.info(f"  - Converting PIL → Torch GPU tensor (preprocessed)")
        # Use backend's preprocessor to get properly formatted tensor
        if hasattr(backend, 'preprocess') and backend.preprocess is not None:  # type: ignore[attr-defined]
            preprocessed = backend.preprocess(test_image.convert("RGB"))  # type: ignore[attr-defined]
            return preprocessed.cuda() if not preprocessed.is_cuda else preprocessed
        else:
            # Fallback: basic conversion
            np_image = np.array(test_image).astype(np.float32) / 255.0
            if np_image.ndim == 3:
                np_image = np_image.transpose(2, 0, 1)  # HWC → CHW
            return torch.from_numpy(np_image).cuda()
    
    else:
        logger.warning(f"  - Unknown format '{native_format}', defaulting to PIL")
        return test_image


def test_image_backend_encode(
    backend: ClipImageRuntimeBase,
    test_image: Image.Image,
    backend_name: str
) -> Tuple[bool, torch.Tensor]:
    """
    Test the image backend encode_image method.
    
    Args:
        backend: Image backend instance to test
        test_image: Input test image (PIL format)
        backend_name: Name of the backend for logging
        
    Returns:
        Tuple of (success, image_features)
    """
    logger.info(f"Testing {backend_name} encode_image()...")
    
    # Prepare image in backend's preferred format
    prepared_image = prepare_image_for_backend(test_image, backend)
    
    try:
        image_features = backend.encode_image(prepared_image)  # type: ignore[arg-type]  # Testing format flexibility
        logger.info(f"✓ encode_image() successful")
        logger.info(f"  - Feature shape: {image_features.shape}")
        logger.info(f"  - Feature dtype: {image_features.dtype}")
        logger.info(f"  - Feature norm: {image_features.norm(dim=-1).item():.6f}")
        logger.info(f"  - Feature range: [{image_features.min().item():.4f}, {image_features.max().item():.4f}]")
        return True, image_features
    except Exception as e:
        logger.error(f"✗ encode_image() failed: {e}", exc_info=True)
        return False, torch.tensor([])


def test_text_backend_encode(
    backend: ClipTextRuntimeBase,
    text_prompts: List[str],
    backend_name: str
) -> Tuple[bool, torch.Tensor]:
    """
    Test the text backend encode_text method.
    
    Args:
        backend: Text backend instance to test
        text_prompts: List of text prompts to encode
        backend_name: Name of the backend for logging
        
    Returns:
        Tuple of (success, text_features)
    """
    logger.info(f"Testing {backend_name} encode_text()...")
    logger.info(f"  - Text prompts: {text_prompts}")
    
    try:
        text_features = backend.encode_text(text_prompts)
        logger.info(f"✓ encode_text() successful")
        logger.info(f"  - Feature shape: {text_features.shape}")
        logger.info(f"  - Feature dtype: {text_features.dtype}")
        logger.info(f"  - Feature norms: {text_features.norm(dim=-1).tolist()}")
        logger.info(f"  - Feature range: [{text_features.min().item():.4f}, {text_features.max().item():.4f}]")
        return True, text_features
    except Exception as e:
        logger.error(f"✗ encode_text() failed: {e}", exc_info=True)
        return False, torch.tensor([])


def test_similarity(
    image_features: torch.Tensor,
    text_features: torch.Tensor,
    text_prompts: List[str]
) -> bool:
    """
    Test image-text similarity computation.
    
    Args:
        image_features: Encoded image features
        text_features: Encoded text features
        text_prompts: List of text prompts (for display)
        
    Returns:
        True if similarity computation successful
    """
    logger.info(f"Testing image-text similarity computation...")
    
    try:
        # Compute similarity scores (scaled by 100 as per CLIP convention)
        similarity = (100.0 * image_features @ text_features.T).cpu()
        
        logger.info(f"✓ Similarity computation successful")
        logger.info(f"  - Similarity scores:")
        
        for i, (prompt, score) in enumerate(zip(text_prompts, similarity.flatten().tolist())):
            logger.info(f"    {i+1}. '{prompt}': {score:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Similarity computation failed: {e}", exc_info=True)
        return False


def test_image_backend_full(
    backend_class,
    backend_name: str,
    config: VLMChatConfig,
    test_image: Image.Image
) -> Tuple[bool, torch.Tensor]:
    """
    Run full test suite on image backend.
    
    Args:
        backend_class: Backend class to instantiate
        backend_name: Name of the backend for logging
        config: Application configuration
        test_image: Test image to use
        
    Returns:
        Tuple of (all_tests_passed, image_features)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {backend_name} Image Backend")
    logger.info(f"{'='*60}")
    
    # Initialize backend
    try:
        backend = backend_class(config)
        logger.info(f"✓ Backend instantiated")
    except Exception as e:
        logger.error(f"✗ Failed to instantiate backend: {e}", exc_info=True)
        return False, torch.tensor([])
    
    # Test availability
    if not test_backend_availability(backend, backend_name):
        return False, torch.tensor([])
    
    # Test encode_image
    success, image_features = test_image_backend_encode(backend, test_image, backend_name)
    if not success:
        return False, torch.tensor([])
    
    logger.info(f"\n✓ All {backend_name} image backend tests passed!")
    return True, image_features


def test_text_backend_full(
    backend_class,
    backend_name: str,
    config: VLMChatConfig,
    text_prompts: List[str]
) -> Tuple[bool, torch.Tensor]:
    """
    Run full test suite on text backend.
    
    Args:
        backend_class: Backend class to instantiate
        backend_name: Name of the backend for logging
        config: Application configuration
        text_prompts: Text prompts to test
        
    Returns:
        Tuple of (all_tests_passed, text_features)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {backend_name} Text Backend")
    logger.info(f"{'='*60}")
    
    # Initialize backend
    try:
        backend = backend_class(config)
        logger.info(f"✓ Backend instantiated")
    except Exception as e:
        logger.error(f"✗ Failed to instantiate backend: {e}", exc_info=True)
        return False, torch.tensor([])
    
    # Test availability
    if not test_backend_availability(backend, backend_name):
        return False, torch.tensor([])
    
    # Test encode_text
    success, text_features = test_text_backend_encode(backend, text_prompts, backend_name)
    if not success:
        return False, torch.tensor([])
    
    logger.info(f"\n✓ All {backend_name} text backend tests passed!")
    return True, text_features


def test_backend_comparison_image(
    config: VLMChatConfig,
    test_image: Image.Image
) -> bool:
    """
    Compare TensorRT and OpenCLIP image backends to verify consistency.
    
    Args:
        config: Application configuration
        test_image: Test image to encode
        
    Returns:
        True if backends produce similar results
    """
    logger.info(f"\n{'='*60}")
    logger.info("Comparing TensorRT vs OpenCLIP Image Backends")
    logger.info(f"{'='*60}")
    
    if not TENSORRT_AVAILABLE:
        logger.warning("✗ TensorRT not available, skipping comparison")
        return False
    
    try:
        # Initialize both backends
        trt_backend = TensorRTImageBackend(config)  # type: ignore[possibly-unbound]
        openclip_backend = OpenClipImageBackend(config)
        
        if not trt_backend.is_available:
            logger.warning("✗ TensorRT image backend not available")
            return False
        
        if not openclip_backend.is_available:
            logger.warning("✗ OpenCLIP image backend not available")
            return False
        
        logger.info("✓ Both backends loaded successfully")
        
        # Both backends accept PIL - let them do their own preprocessing
        logger.info("Encoding image with TensorRT...")
        logger.info(f"  - TensorRT declares native format: {trt_backend.native_image_format}")
        logger.info(f"  - Providing PIL (backend will preprocess)")
        trt_features = trt_backend.encode_image(test_image)
        
        logger.info("Encoding image with OpenCLIP...")
        logger.info(f"  - OpenCLIP declares native format: {openclip_backend.native_image_format}")
        logger.info(f"  - Providing PIL (backend will preprocess)")
        openclip_features = openclip_backend.encode_image(test_image)
        
        # Compare shapes
        if trt_features.shape != openclip_features.shape:
            logger.error(f"✗ Shape mismatch: TensorRT {trt_features.shape} vs OpenCLIP {openclip_features.shape}")
            return False
        
        logger.info(f"✓ Shapes match: {trt_features.shape}")
        
        # Compute cosine similarity
        similarity = (trt_features @ openclip_features.T).item()
        logger.info(f"  Cosine similarity: {similarity:.6f}")
        
        # Compute L2 distance
        l2_distance = torch.norm(trt_features - openclip_features).item()
        logger.info(f"  L2 distance: {l2_distance:.6f}")
        
        # Check if results are reasonably similar (cosine similarity > 0.99)
        if similarity > 0.99:
            logger.info(f"✓ Backends produce consistent results (similarity: {similarity:.6f})")
            return True
        else:
            logger.warning(f"⚠ Backends have low similarity: {similarity:.6f}")
            return False
            
    except Exception as e:
        logger.error(f"✗ Backend comparison failed: {e}", exc_info=True)
        return False


def test_backend_comparison_text(
    config: VLMChatConfig,
    text_prompts: List[str]
) -> bool:
    """
    Compare TensorRT and OpenCLIP text backends to verify consistency.
    
    Args:
        config: Application configuration
        text_prompts: Text prompts to encode
        
    Returns:
        True if backends produce similar results
    """
    logger.info(f"\n{'='*60}")
    logger.info("Comparing TensorRT vs OpenCLIP Text Backends")
    logger.info(f"{'='*60}")
    
    if not TENSORRT_AVAILABLE:
        logger.warning("✗ TensorRT not available, skipping comparison")
        return False
    
    try:
        # Initialize both backends
        trt_backend = TensorRTTextBackend(config)  # type: ignore[possibly-unbound]
        openclip_backend = OpenClipTextBackend(config)
        
        if not trt_backend.is_available:
            logger.warning("✗ TensorRT text backend not available")
            return False
        
        if not openclip_backend.is_available:
            logger.warning("✗ OpenCLIP text backend not available")
            return False
        
        logger.info("✓ Both backends loaded successfully")
        
        # Encode with both backends
        logger.info(f"Encoding {len(text_prompts)} prompts with TensorRT...")
        trt_features = trt_backend.encode_text(text_prompts)
        
        logger.info(f"Encoding {len(text_prompts)} prompts with OpenCLIP...")
        openclip_features = openclip_backend.encode_text(text_prompts)
        
        # Compare shapes
        if trt_features.shape != openclip_features.shape:
            logger.error(f"✗ Shape mismatch: TensorRT {trt_features.shape} vs OpenCLIP {openclip_features.shape}")
            return False
        
        logger.info(f"✓ Shapes match: {trt_features.shape}")
        
        # Compute per-prompt cosine similarity
        all_similar = True
        for i, prompt in enumerate(text_prompts):
            similarity = (trt_features[i] @ openclip_features[i]).item()
            l2_distance = torch.norm(trt_features[i] - openclip_features[i]).item()
            
            logger.info(f"  Prompt {i+1} ('{prompt}'):")
            logger.info(f"    Cosine similarity: {similarity:.6f}")
            logger.info(f"    L2 distance: {l2_distance:.6f}")
            
            if similarity <= 0.99:
                logger.warning(f"    ⚠ Low similarity for prompt {i+1}")
                all_similar = False
        
        if all_similar:
            logger.info(f"✓ Backends produce consistent results for all prompts")
            return True
        else:
            logger.warning(f"⚠ Some prompts have low similarity")
            return False
            
    except Exception as e:
        logger.error(f"✗ Backend comparison failed: {e}", exc_info=True)
        return False


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test CLIP backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--backend',
        choices=['image', 'text', 'all'],
        default='all',
        help='Which backend to test (default: all)'
    )
    
    parser.add_argument(
        '--implementation',
        choices=['openclip', 'tensorrt'],
        default='openclip',
        help='Which implementation to use (default: openclip)'
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
        help='Path to test image (default: trail-riders.jpg in script directory)'
    )
    
    parser.add_argument(
        '--prompts',
        nargs='+',
        default=["a horse", "a person riding a horse", "a man wearing a hat"],
        help='Text prompts to test (default: ["a horse", "a person riding a horse", "a man wearing a hat"])'
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
    
    logger.info("CLIP Backend Test Suite")
    logger.info(f"Project root: {PROJECT_ROOT}")
    
    # Check TensorRT availability if requested
    if args.implementation == 'tensorrt' and not TENSORRT_AVAILABLE:
        logger.error("✗ TensorRT implementation requested but not available")
        logger.error("  Install tensorrt and pycuda to use TensorRT backends")
        return 1
    
    # Select backend classes based on implementation
    if args.implementation == 'openclip':
        image_backend_class = OpenClipImageBackend
        text_backend_class = OpenClipTextBackend
        impl_name = "OpenCLIP"
    else:  # tensorrt
        image_backend_class = TensorRTImageBackend  # type: ignore[possibly-unbound]
        text_backend_class = TensorRTTextBackend  # type: ignore[possibly-unbound]
        impl_name = "TensorRT"
    
    logger.info(f"Testing {impl_name} implementation")
    
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
    
    # Test results
    image_passed = False
    text_passed = False
    image_features = torch.tensor([])
    text_features = torch.tensor([])
    test_image = None
    
    # Test image backend
    if args.backend in ('image', 'all'):
        # Determine image path
        if args.image:
            image_path = args.image
        else:
            image_path = SCRIPT_DIR / "trail-riders.jpg"
        
        # Load test image
        try:
            test_image = load_test_image(image_path)
        except Exception as e:
            logger.error(f"✗ Failed to load test image: {e}", exc_info=True)
            return 1
        
        # Run image backend test
        try:
            image_passed, image_features = test_image_backend_full(
                image_backend_class,
                impl_name,
                config,
                test_image
            )
        except Exception as e:
            logger.error(f"Image backend test crashed: {e}", exc_info=True)
            image_passed = False
    
    # Test text backend
    if args.backend in ('text', 'all'):
        try:
            text_passed, text_features = test_text_backend_full(
                text_backend_class,
                impl_name,
                config,
                args.prompts
            )
        except Exception as e:
            logger.error(f"Text backend test crashed: {e}", exc_info=True)
            text_passed = False
    
    # Test similarity if both backends passed
    similarity_passed = False
    if args.backend == 'all' and image_passed and text_passed:
        logger.info(f"\n{'='*60}")
        logger.info("Testing Image-Text Similarity")
        logger.info(f"{'='*60}")
        
        similarity_passed = test_similarity(image_features, text_features, args.prompts)
    
    # Run comparison tests if TensorRT is available and requested
    image_comparison_passed = False
    text_comparison_passed = False
    
    if TENSORRT_AVAILABLE and args.backend in ('image', 'all') and test_image is not None:
        try:
            image_comparison_passed = test_backend_comparison_image(config, test_image)
        except Exception as e:
            logger.error(f"Image comparison test crashed: {e}", exc_info=True)
            image_comparison_passed = False
    
    if TENSORRT_AVAILABLE and args.backend in ('text', 'all'):
        try:
            text_comparison_passed = test_backend_comparison_text(config, args.prompts)
        except Exception as e:
            logger.error(f"Text comparison test crashed: {e}", exc_info=True)
            text_comparison_passed = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    
    if args.backend in ('image', 'all'):
        status = "✓ PASSED" if image_passed else "✗ FAILED"
        logger.info(f"Image Backend:       {status}")
        
        if TENSORRT_AVAILABLE:
            status = "✓ PASSED" if image_comparison_passed else "✗ FAILED"
            logger.info(f"Image Comparison:    {status}")
    
    if args.backend in ('text', 'all'):
        status = "✓ PASSED" if text_passed else "✗ FAILED"
        logger.info(f"Text Backend:        {status}")
        
        if TENSORRT_AVAILABLE:
            status = "✓ PASSED" if text_comparison_passed else "✗ FAILED"
            logger.info(f"Text Comparison:     {status}")
    
    if args.backend == 'all' and image_passed and text_passed:
        status = "✓ PASSED" if similarity_passed else "✗ FAILED"
        logger.info(f"Similarity Test:     {status}")
    
    # Return exit code
    if args.backend == 'image':
        return 0 if image_passed else 1
    elif args.backend == 'text':
        return 0 if text_passed else 1
    else:  # all
        if image_passed and text_passed:
            logger.info("\n✓ All CLIP backends are working!")
            return 0
        else:
            logger.error("\n✗ Some backends failed!")
            return 1


if __name__ == "__main__":
    sys.exit(main())
