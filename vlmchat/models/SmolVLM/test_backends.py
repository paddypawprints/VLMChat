"""
Test script for SmolVLM backends (PyTorch and ONNX).

This script tests backend implementations to verify they work correctly
and produce consistent results.

Run with: 
    cd /path/to/VLMChat
    python src/models/SmolVLM/test_backends.py [options]

Or from src directory:
    cd /path/to/VLMChat/src
    python -m models.SmolVLM.test_backends [options]

Options:
    --backend {pytorch,onnx,all}         Which backend to test (default: all)
    --config PATH                        Path to config file (default: config.json)
    --image PATH                         Path to test image (default: captures/camera0_20251014_180719.jpg)
    --prompt TEXT                        Text prompt to test (default: "What do you see?")
    --max-tokens N                       Maximum tokens to generate (default: 20)
    --verbose                            Enable verbose logging
"""

import sys
import logging
import argparse
import time
from pathlib import Path
from typing import Tuple, List

# Set up project root and add src to path
SCRIPT_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
SRC_DIR = PROJECT_ROOT / "src"

# Add both src and project root to path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from PIL import Image

# Try importing with proper path setup
try:
    from utils.config import VLMChatConfig
    from models.SmolVLM.runtime_base import SmolVLMRuntimeBase
except ImportError:
    # Try alternative import
    from vlmchat.utils.config import VLMChatConfig  # type: ignore[no-redef]
    from vlmchat.models.SmolVLM.runtime_base import SmolVLMRuntimeBase  # type: ignore[no-redef]

# Try importing backends
try:
    from models.SmolVLM.pytorch_backend import PyTorchBackend
    PYTORCH_AVAILABLE = True
except ImportError:
    try:
        from vlmchat.models.SmolVLM.pytorch_backend import PyTorchBackend  # type: ignore[no-redef]
        PYTORCH_AVAILABLE = True
    except ImportError as e:
        PYTORCH_AVAILABLE = False
        PYTORCH_ERROR = str(e)

try:
    from models.SmolVLM.onnx_backend import OnnxBackend
    ONNX_AVAILABLE = True
except ImportError:
    try:
        from vlmchat.models.SmolVLM.onnx_backend import OnnxBackend  # type: ignore[no-redef]
        ONNX_AVAILABLE = True
    except ImportError as e:
        ONNX_AVAILABLE = False
        ONNX_ERROR = str(e)

logger = logging.getLogger(__name__)


def load_test_image(image_path: Path, max_size: int = None) -> Image.Image:
    """
    Load a test image from the specified path.
    
    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height). Image will be resized if larger.
        
    Returns:
        PIL Image
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")
    
    image = Image.open(image_path)
    original_size = image.size
    
    # Resize if max_size specified and image exceeds it
    if max_size is not None:
        width, height = image.size
        if width > max_size or height > max_size:
            # Calculate new size maintaining aspect ratio
            if width > height:
                new_width = max_size
                new_height = int(height * (max_size / width))
            else:
                new_height = max_size
                new_width = int(width * (max_size / height))
            
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            logger.info(f"Resized image: {original_size} → {image.size}")
    
    logger.info(f"Loaded test image: {image.size} from {image_path}")
    return image


def test_backend_availability(
    backend: SmolVLMRuntimeBase, 
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


def test_backend_generate(
    backend: SmolVLMRuntimeBase,
    test_image: Image.Image,
    prompt: str,
    max_tokens: int,
    backend_name: str,
    run_number: int = 1
) -> Tuple[bool, str, float]:
    """
    Test the backend generate method.
    
    Args:
        backend: Backend instance to test
        test_image: Input test image (PIL format)
        prompt: Text prompt for generation
        max_tokens: Maximum number of tokens to generate
        backend_name: Name of the backend for logging
        run_number: Which run this is (for logging)
        
    Returns:
        Tuple of (success, generated_text, generation_time)
    """
    logger.info(f"Testing {backend_name} generate() [Run {run_number}]...")
    logger.info(f"  - Prompt: '{prompt}'")
    logger.info(f"  - Max tokens: {max_tokens}")
    
    try:
        # Format messages properly for SmolVLM
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Prepare inputs using backend's prepare_inputs method
        inputs = backend.prepare_inputs(messages, [test_image])
        
        start_time = time.time()
        generated_tokens = []
        
        # Track timing for different stages
        token_times = []
        last_time = start_time
        
        # Collect generated tokens
        for token in backend.generate(inputs, max_new_tokens=max_tokens):
            current_time = time.time()
            token_time = current_time - last_time
            token_times.append(token_time)
            last_time = current_time
            
            generated_tokens.append(token)
            if len(generated_tokens) <= 5 or len(generated_tokens) % 10 == 0:
                logger.info(f"  Token {len(generated_tokens)}: '{token}' ({token_time:.3f}s)")
            elif len(generated_tokens) <= 10:
                logger.info(f"  Token {len(generated_tokens)}: '{token}' ({token_time:.3f}s)")
        
        elapsed = time.time() - start_time
        generated_text = "".join(generated_tokens)
        
        logger.info(f"✓ generate() successful")
        logger.info(f"  - Generated {len(generated_tokens)} tokens in {elapsed:.2f}s")
        logger.info(f"  - Average speed: {elapsed/len(generated_tokens):.3f}s per token")
        
        # Analyze token timing
        if len(token_times) > 1:
            first_token_time = token_times[0]
            subsequent_times = token_times[1:]
            avg_subsequent = sum(subsequent_times) / len(subsequent_times)
            logger.info(f"  - First token (includes vision): {first_token_time:.3f}s")
            logger.info(f"  - Avg subsequent tokens: {avg_subsequent:.3f}s")
            logger.info(f"  - Min token time: {min(subsequent_times):.3f}s")
            logger.info(f"  - Max token time: {max(subsequent_times):.3f}s")
        
        logger.info(f"  - Full text: '{generated_text}'")
        
        return True, generated_text, elapsed
        
    except Exception as e:
        logger.error(f"✗ generate() failed: {e}", exc_info=True)
        return False, "", 0.0


def test_backend_full(
    backend_class,
    backend_name: str,
    config: VLMChatConfig,
    test_image: Image.Image,
    prompt: str,
    max_tokens: int,
    num_runs: int = 1
) -> Tuple[bool, List[str], List[float]]:
    """
    Run full test suite on backend.
    
    Args:
        backend_class: Backend class to instantiate
        backend_name: Name of the backend for logging
        config: Application configuration
        test_image: Test image to use
        prompt: Text prompt for generation
        max_tokens: Maximum tokens to generate
        num_runs: Number of times to run inference
        
    Returns:
        Tuple of (all_tests_passed, list_of_generated_texts, list_of_generation_times)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Testing {backend_name} Backend ({num_runs} run{'s' if num_runs > 1 else ''})")
    logger.info(f"{'='*60}")
    
    # Initialize backend
    try:
        logger.info(f"Initializing {backend_name} backend...")
        start_init = time.time()
        backend = backend_class(config)
        elapsed_init = time.time() - start_init
        logger.info(f"✓ Backend instantiated in {elapsed_init:.2f}s")
        
        # Log backend details
        if hasattr(backend, '_device'):
            logger.info(f"  - Device: {backend._device}")
        if hasattr(backend, '_eos_token_id'):
            logger.info(f"  - EOS tokens: {backend._eos_token_id}")
            
    except Exception as e:
        logger.error(f"✗ Failed to instantiate backend: {e}", exc_info=True)
        return False, [], []
    
    # Test availability
    if not test_backend_availability(backend, backend_name):
        return False, [], []
    
    # Run generate multiple times
    all_texts = []
    all_times = []
    all_success = True
    
    for run in range(1, num_runs + 1):
        success, generated_text, gen_time = test_backend_generate(
            backend, test_image, prompt, max_tokens, backend_name, run
        )
        if not success:
            all_success = False
            break
        all_texts.append(generated_text)
        all_times.append(gen_time)
    
    if not all_success:
        return False, [], []
    
    # Log summary if multiple runs
    if num_runs > 1:
        logger.info(f"\n--- Summary of {num_runs} runs ---")
        avg_time = sum(all_times) / len(all_times)
        min_time = min(all_times)
        max_time = max(all_times)
        logger.info(f"  - Average time: {avg_time:.2f}s ({avg_time/max_tokens:.2f}s/token)")
        logger.info(f"  - Min time: {min_time:.2f}s")
        logger.info(f"  - Max time: {max_time:.2f}s")
        
        # Check consistency
        if len(set(all_texts)) == 1:
            logger.info(f"  - ✓ All runs produced identical output")
        else:
            logger.warning(f"  - ⚠ Outputs varied across runs:")
            for i, text in enumerate(all_texts, 1):
                logger.warning(f"    Run {i}: '{text}'")
    
    logger.info(f"\n✓ All {backend_name} backend tests passed!")
    return True, all_texts, all_times


def test_backend_comparison(
    config: VLMChatConfig,
    test_image: Image.Image,
    prompt: str,
    max_tokens: int
) -> bool:
    """
    Compare PyTorch and ONNX backends to verify consistency.
    
    Args:
        config: Application configuration
        test_image: Test image to use
        prompt: Text prompt for generation
        max_tokens: Maximum tokens to generate
        
    Returns:
        True if backends produce similar results
    """
    logger.info(f"\n{'='*60}")
    logger.info("Comparing PyTorch vs ONNX Backends")
    logger.info(f"{'='*60}")
    
    if not PYTORCH_AVAILABLE:
        logger.warning("✗ PyTorch backend not available, skipping comparison")
        return False
    
    if not ONNX_AVAILABLE:
        logger.warning("✗ ONNX backend not available, skipping comparison")
        return False
    
    try:
        # Initialize both backends
        pytorch_backend = PyTorchBackend(config)  # type: ignore[possibly-unbound]
        onnx_backend = OnnxBackend(config)  # type: ignore[possibly-unbound]
        
        if not pytorch_backend.is_available:
            logger.warning("✗ PyTorch backend not available")
            return False
        
        if not onnx_backend.is_available:
            logger.warning("✗ ONNX backend not available")
            return False
        
        logger.info("✓ Both backends loaded successfully")
        
        # Format messages
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Prepare inputs
        pytorch_inputs = pytorch_backend.prepare_inputs(messages, [test_image])
        onnx_inputs = onnx_backend.prepare_inputs(messages, [test_image])
        
        # Generate with PyTorch
        logger.info("Generating with PyTorch backend...")
        pytorch_tokens = []
        for token in pytorch_backend.generate(pytorch_inputs, max_new_tokens=max_tokens):
            pytorch_tokens.append(token)
        pytorch_text = "".join(pytorch_tokens)
        logger.info(f"  PyTorch: '{pytorch_text}'")
        
        # Generate with ONNX
        logger.info("Generating with ONNX backend...")
        onnx_tokens = []
        for token in onnx_backend.generate(onnx_inputs, max_new_tokens=max_tokens):
            onnx_tokens.append(token)
        onnx_text = "".join(onnx_tokens)
        logger.info(f"  ONNX: '{onnx_text}'")
        
        # Compare outputs
        if pytorch_text == onnx_text:
            logger.info(f"✓ Backends produce identical outputs")
            return True
        else:
            # Check token-by-token
            min_len = min(len(pytorch_tokens), len(onnx_tokens))
            matching = sum(1 for i in range(min_len) if pytorch_tokens[i] == onnx_tokens[i])
            match_pct = 100.0 * matching / max(len(pytorch_tokens), len(onnx_tokens))
            
            logger.warning(f"⚠ Backends produce different outputs")
            logger.warning(f"  Token match: {matching}/{max(len(pytorch_tokens), len(onnx_tokens))} ({match_pct:.1f}%)")
            logger.warning(f"  PyTorch ({len(pytorch_tokens)} tokens): '{pytorch_text}'")
            logger.warning(f"  ONNX ({len(onnx_tokens)} tokens): '{onnx_text}'")
            
            # Consider it a pass if at least 80% of tokens match
            if match_pct >= 80.0:
                logger.info(f"✓ Backends are reasonably consistent ({match_pct:.1f}% match)")
                return True
            else:
                logger.error(f"✗ Backends are too different ({match_pct:.1f}% match)")
                return False
            
    except Exception as e:
        logger.error(f"✗ Backend comparison failed: {e}", exc_info=True)
        return False


def main():
    """Main entry point for testing SmolVLM backends."""
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Test SmolVLM backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        '--backend',
        choices=['pytorch', 'onnx', 'all'],
        default='all',
        help='Which backend to test (default: all)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=PROJECT_ROOT / 'config.json',
        help='Path to config file (default: config.json)'
    )
    parser.add_argument(
        '--image',
        type=Path,
        default=PROJECT_ROOT / 'captures' / 'camera0_20251014_180719.jpg',
        help='Path to test image'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default='What do you see?',
        help='Text prompt for generation (default: "What do you see?")'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=20,
        help='Maximum tokens to generate (default: 20)'
    )
    parser.add_argument(
        '--runs',
        type=int,
        default=1,
        help='Number of times to run inference (default: 1)'
    )
    parser.add_argument(
        '--max-image-size',
        type=int,
        default=None,
        help='Maximum image dimension (width or height). Image will be resized if larger (default: no resize)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(levelname)s: %(message)s'
    )
    
    # Suppress some noisy loggers
    logging.getLogger('PIL').setLevel(logging.WARNING)
    
    # Load config
    try:
        import json
        with open(args.config) as f:
            config = VLMChatConfig(json.load(f))
        logger.info(f"✓ Config loaded from {args.config}")
    except Exception as e:
        logger.error(f"✗ Failed to load config: {e}")
        return 1
    
    # Load test image
    try:
        test_image = load_test_image(args.image, args.max_image_size)
    except Exception as e:
        logger.error(f"✗ Failed to load test image: {e}")
        return 1
    
    # Check backend availability
    logger.info(f"\n{'='*60}")
    logger.info("Backend Availability")
    logger.info(f"{'='*60}")
    logger.info(f"PyTorch: {'✓ Available' if PYTORCH_AVAILABLE else '✗ Not available'}")
    if not PYTORCH_AVAILABLE:
        logger.info(f"  Error: {PYTORCH_ERROR}")
    logger.info(f"ONNX: {'✓ Available' if ONNX_AVAILABLE else '✗ Not available'}")
    if not ONNX_AVAILABLE:
        logger.info(f"  Error: {ONNX_ERROR}")
    
    # Determine which backend to use
    if args.backend == 'pytorch':
        if not PYTORCH_AVAILABLE:
            logger.error("✗ PyTorch backend requested but not available")
            return 1
        backend_class = PyTorchBackend  # type: ignore[possibly-unbound]
        backend_name = "PyTorch"
    elif args.backend == 'onnx':
        if not ONNX_AVAILABLE:
            logger.error("✗ ONNX backend requested but not available")
            return 1
        backend_class = OnnxBackend  # type: ignore[possibly-unbound]
        backend_name = "ONNX"
    else:  # all
        backend_class = None
        backend_name = None
    
    # Test results
    pytorch_passed = False
    onnx_passed = False
    pytorch_texts = []
    onnx_texts = []
    pytorch_times = []
    onnx_times = []
    
    # Test PyTorch backend
    if args.backend in ('pytorch', 'all') and PYTORCH_AVAILABLE:
        try:
            pytorch_passed, pytorch_texts, pytorch_times = test_backend_full(
                PyTorchBackend,  # type: ignore[possibly-unbound]
                "PyTorch",
                config,
                test_image,
                args.prompt,
                args.max_tokens,
                args.runs
            )
        except Exception as e:
            logger.error(f"PyTorch backend test crashed: {e}", exc_info=True)
            pytorch_passed = False
    
    # Test ONNX backend
    if args.backend in ('onnx', 'all') and ONNX_AVAILABLE:
        try:
            onnx_passed, onnx_texts, onnx_times = test_backend_full(
                OnnxBackend,  # type: ignore[possibly-unbound]
                "ONNX",
                config,
                test_image,
                args.prompt,
                args.max_tokens,
                args.runs
            )
        except Exception as e:
            logger.error(f"ONNX backend test crashed: {e}", exc_info=True)
            onnx_passed = False
    
    # Run comparison test if both backends available
    comparison_passed = False
    if args.backend == 'all' and PYTORCH_AVAILABLE and ONNX_AVAILABLE:
        try:
            comparison_passed = test_backend_comparison(
                config, test_image, args.prompt, args.max_tokens
            )
        except Exception as e:
            logger.error(f"Comparison test crashed: {e}", exc_info=True)
            comparison_passed = False
    
    # Summary
    logger.info(f"\n{'='*60}")
    logger.info("Test Summary")
    logger.info(f"{'='*60}")
    
    if args.backend in ('pytorch', 'all') and PYTORCH_AVAILABLE:
        status = "✓ PASSED" if pytorch_passed else "✗ FAILED"
        logger.info(f"PyTorch Backend:     {status}")
        if pytorch_passed and pytorch_texts:
            if args.runs == 1:
                logger.info(f"  Generated: '{pytorch_texts[0]}'")
                logger.info(f"  Time: {pytorch_times[0]:.2f}s ({pytorch_times[0]/args.max_tokens:.2f}s/token)")
            else:
                avg_time = sum(pytorch_times) / len(pytorch_times)
                logger.info(f"  Average time: {avg_time:.2f}s ({avg_time/args.max_tokens:.2f}s/token)")
                logger.info(f"  First output: '{pytorch_texts[0]}'")
    
    if args.backend in ('onnx', 'all') and ONNX_AVAILABLE:
        status = "✓ PASSED" if onnx_passed else "✗ FAILED"
        logger.info(f"ONNX Backend:        {status}")
        if onnx_passed and onnx_texts:
            if args.runs == 1:
                logger.info(f"  Generated: '{onnx_texts[0]}'")
                logger.info(f"  Time: {onnx_times[0]:.2f}s ({onnx_times[0]/args.max_tokens:.2f}s/token)")
            else:
                avg_time = sum(onnx_times) / len(onnx_times)
                logger.info(f"  Average time: {avg_time:.2f}s ({avg_time/args.max_tokens:.2f}s/token)")
                logger.info(f"  First output: '{onnx_texts[0]}'")
    
    if args.backend == 'all' and PYTORCH_AVAILABLE and ONNX_AVAILABLE:
        status = "✓ PASSED" if comparison_passed else "✗ FAILED"
        logger.info(f"Backend Comparison:  {status}")
    
    # Return exit code
    if args.backend == 'pytorch':
        return 0 if pytorch_passed else 1
    elif args.backend == 'onnx':
        return 0 if onnx_passed else 1
    else:  # all
        if PYTORCH_AVAILABLE and ONNX_AVAILABLE:
            if pytorch_passed and onnx_passed:
                logger.info("\n✓ All SmolVLM backends are working!")
                return 0
            else:
                logger.error("\n✗ Some backends failed!")
                return 1
        elif ONNX_AVAILABLE and onnx_passed:
            logger.info("\n✓ ONNX backend is working!")
            return 0
        elif PYTORCH_AVAILABLE and pytorch_passed:
            logger.info("\n✓ PyTorch backend is working!")
            return 0
        else:
            logger.error("\n✗ No backends available or all failed!")
            return 1


if __name__ == "__main__":
    sys.exit(main())
