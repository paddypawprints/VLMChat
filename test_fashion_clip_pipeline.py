#!/usr/bin/env python3
"""
Test FashionClip pipeline tasks.

Demonstrates FashionClip text and vision encoders integrated with the pipeline system.
"""

import sys
from pathlib import Path
import logging

# Add src to path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from utils.config import VLMChatConfig
from models.FashionClip import FashionClipModel
from pipeline.task_base import Context, ContextDataType
from pipeline.tasks.fashion_clip_text_encoder_task import FashionClipTextEncoderTask
from pipeline.tasks.fashion_clip_vision_task import FashionClipVisionTask
from object_detector.detection_base import Detection
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_fashion_clip_text_encoder():
    """Test FashionClip text encoder task."""
    print("\n" + "="*70)
    print("TEST 1: FashionClip Text Encoder Task")
    print("="*70)
    
    # Initialize model
    config = VLMChatConfig()
    fashion_clip = FashionClipModel(config)
    
    # Create task
    text_encoder = FashionClipTextEncoderTask(
        prompts=["red dress", "blue jeans", "leather jacket"],
        fashion_clip_model=fashion_clip,
        task_id="fashion_text_encoder"
    )
    
    # Create context
    ctx = Context()
    
    # Run task
    logger.info("Encoding fashion prompts...")
    result_ctx = text_encoder.run(ctx)
    
    # Verify results
    assert ContextDataType.PROMPT_EMBEDDINGS in result_ctx.data
    embeddings_data = result_ctx.data[ContextDataType.PROMPT_EMBEDDINGS]
    
    assert 'prompts' in embeddings_data
    assert 'embeddings' in embeddings_data
    assert len(embeddings_data['prompts']) == 3
    
    embeddings = embeddings_data['embeddings']
    print(f"\nResults:")
    print(f"  Prompts: {embeddings_data['prompts']}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Embeddings dtype: {embeddings.dtype}")
    print(f"  Expected: (3, 768) float32")
    
    assert embeddings.shape == (3, 768), f"Expected (3, 768), got {embeddings.shape}"
    assert embeddings.dtype == np.float32
    
    # Check normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"  L2 norms: {norms}")
    assert np.allclose(norms, 1.0, atol=0.01), "Embeddings should be normalized"
    
    print("\n✅ TEST 1 PASSED: FashionClip text encoder generates correct embeddings")
    return True


def test_fashion_clip_vision_task():
    """Test FashionClip vision encoder task."""
    print("\n" + "="*70)
    print("TEST 2: FashionClip Vision Task")
    print("="*70)
    
    # Initialize model
    config = VLMChatConfig()
    fashion_clip = FashionClipModel(config)
    
    # Create task
    vision_encoder = FashionClipVisionTask(
        fashion_clip_model=fashion_clip,
        task_id="fashion_vision_encoder"
    )
    
    # Create test image
    test_image = Image.new('RGB', (640, 480), color='red')
    
    # Create fake detections (fashion items)
    detections = [
        Detection(box=(50, 50, 200, 300), object_category="dress", conf=0.95),
        Detection(box=(300, 100, 450, 400), object_category="shoes", conf=0.89),
        Detection(box=(500, 50, 620, 200), object_category="hat", conf=0.92)
    ]
    
    # Create context
    ctx = Context()
    ctx.data[ContextDataType.IMAGE] = [test_image]
    ctx.data[ContextDataType.DETECTIONS] = detections
    
    # Run task
    logger.info("Encoding fashion detection crops...")
    result_ctx = vision_encoder.run(ctx)
    
    # Verify results
    assert ContextDataType.EMBEDDINGS in result_ctx.data
    embeddings = result_ctx.data[ContextDataType.EMBEDDINGS]
    
    print(f"\nResults:")
    print(f"  Number of detections: {len(detections)}")
    print(f"  Number of embeddings: {len(embeddings)}")
    
    assert len(embeddings) == 3, f"Expected 3 embeddings, got {len(embeddings)}"
    
    # Check each embedding
    for i, emb in enumerate(embeddings):
        print(f"  Embedding {i}: shape={emb.shape}, dtype={emb.dtype}")
        assert emb.shape == (1, 768), f"Expected (1, 768), got {emb.shape}"
        
        # Check normalization
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 0.01, f"Embedding {i} should be normalized, norm={norm}"
    
    print("\n✅ TEST 2 PASSED: FashionClip vision encoder generates correct embeddings")
    return True


def test_fashion_clip_pipeline_integration():
    """Test FashionClip tasks working together in pipeline."""
    print("\n" + "="*70)
    print("TEST 3: FashionClip Pipeline Integration")
    print("="*70)
    
    # Initialize model
    config = VLMChatConfig()
    fashion_clip = FashionClipModel(config)
    
    # Create tasks
    text_encoder = FashionClipTextEncoderTask(
        prompts=["dress", "shoes", "hat", "jacket"],
        fashion_clip_model=fashion_clip,
        task_id="fashion_text"
    )
    
    vision_encoder = FashionClipVisionTask(
        fashion_clip_model=fashion_clip,
        task_id="fashion_vision"
    )
    
    # Create test data
    test_image = Image.new('RGB', (640, 480), color='blue')
    detections = [
        Detection(box=(100, 100, 300, 400), object_category="dress", conf=0.95),
        Detection(box=(350, 150, 500, 450), object_category="shoes", conf=0.88)
    ]
    
    # Create context and run text encoder first
    ctx = Context()
    ctx = text_encoder.run(ctx)
    
    print(f"\nAfter text encoder:")
    print(f"  Prompt embeddings shape: {ctx.data[ContextDataType.PROMPT_EMBEDDINGS]['embeddings'].shape}")
    
    # Add image and detections
    ctx.data[ContextDataType.IMAGE] = [test_image]
    ctx.data[ContextDataType.DETECTIONS] = detections
    
    # Run vision encoder
    ctx = vision_encoder.run(ctx)
    
    print(f"\nAfter vision encoder:")
    print(f"  Vision embeddings: {len(ctx.data[ContextDataType.EMBEDDINGS])} crops")
    
    # Get embeddings
    prompt_embeddings = ctx.data[ContextDataType.PROMPT_EMBEDDINGS]['embeddings']  # (4, 768)
    vision_embeddings = np.array([emb.squeeze() for emb in ctx.data[ContextDataType.EMBEDDINGS]])  # (2, 768)
    
    print(f"\nEmbeddings ready for comparison:")
    print(f"  Prompts: {prompt_embeddings.shape}")
    print(f"  Vision: {vision_embeddings.shape}")
    
    # Compute similarity (like ClipCompareTask would do)
    similarity = np.dot(vision_embeddings, prompt_embeddings.T)
    print(f"\nSimilarity matrix shape: {similarity.shape}")
    print(f"Similarity scores:")
    for i, det in enumerate(detections):
        print(f"  Detection {i} ({det.object_category}):")
        for j, prompt in enumerate(ctx.data[ContextDataType.PROMPT_EMBEDDINGS]['prompts']):
            print(f"    vs '{prompt}': {similarity[i, j]:.4f}")
    
    # Find best matches
    best_matches = np.argmax(similarity, axis=1)
    prompts = ctx.data[ContextDataType.PROMPT_EMBEDDINGS]['prompts']
    
    print(f"\nBest matches:")
    for i, (det, match_idx) in enumerate(zip(detections, best_matches)):
        matched_prompt = prompts[match_idx]
        score = similarity[i, match_idx]
        print(f"  Detection {i} ({det.object_category}) → '{matched_prompt}' (score: {score:.4f})")
    
    print("\n✅ TEST 3 PASSED: FashionClip text and vision encoders work together")
    return True


def main():
    """Run all tests."""
    print("="*70)
    print("FASHIONCLIP PIPELINE TASKS TEST SUITE")
    print("="*70)
    
    tests = [
        ("FashionClip Text Encoder", test_fashion_clip_text_encoder),
        ("FashionClip Vision Task", test_fashion_clip_vision_task),
        ("FashionClip Pipeline Integration", test_fashion_clip_pipeline_integration)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"\n❌ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"\n❌ {name} FAILED with exception:")
            print(f"   {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n✅ ALL TESTS PASSED")
        return 0
    else:
        print(f"\n❌ {failed} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
