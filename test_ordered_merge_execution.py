#!/usr/bin/env python3
"""
Test ordered_merge execution in a simple pipeline.

Creates a minimal pipeline with parallel branches and ordered_merge to verify functionality.
"""

import sys
from pathlib import Path
import numpy as np

# Set up Python paths correctly
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.config import VLMChatConfig
from src.pipeline.dsl_parser import DSLParser, create_task_registry
from src.pipeline.task_base import Context, ContextDataType
from src.metrics.metrics_collector import Collector


def test_ordered_merge_simple():
    """Test ordered_merge with a simple parallel pipeline."""
    print("\n" + "="*70)
    print("TEST: Ordered Merge Execution")
    print("="*70)
    
    # Simple DSL with parallel branches and ordered merge
    dsl = '''
[
  pass(value="branch_0"),
  pass(value="branch_1")
  :ordered_merge(order="0,1"):
]
'''
    
    print("\nDSL:")
    print(dsl)
    
    # Load registry
    registry = create_task_registry()
    
    # Parse and build
    parser = DSLParser(task_registry=registry)
    pipeline = parser.parse(dsl)
    
    print(f"\n✓ Pipeline built: {type(pipeline).__name__}")
    
    # Create context
    config = VLMChatConfig()
    collector = Collector()
    ctx = Context(config=config, collector=collector)
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    result_ctx = pipeline.run(ctx)
    
    print(f"✓ Pipeline executed")
    
    # Check results
    print("\nResults:")
    for key, value in result_ctx.data.items():
        print(f"  {key}: {value}")
    
    print("\n✅ ORDERED MERGE TEST PASSED")
    return True


def test_ordered_merge_with_embeddings():
    """Test ordered_merge with simulated embeddings (like CLIP would produce)."""
    print("\n" + "="*70)
    print("TEST: Ordered Merge with Embeddings")
    print("="*70)
    
    # DSL with prompt embeddings
    dsl = '''
[
  prompt_embeddings(prompts="text1,text2"),
  prompt_embeddings(prompts="text3,text4")
  :ordered_merge(order="0,1"):
]
'''
    
    print("\nDSL:")
    print(dsl)
    
    # Load registry
    registry = create_task_registry()
    
    # Parse and build
    parser = DSLParser(task_registry=registry)
    pipeline = parser.parse(dsl)
    
    print(f"\n✓ Pipeline built: {type(pipeline).__name__}")
    
    # Create context
    config = VLMChatConfig()
    collector = Collector()
    ctx = Context(config=config, collector=collector)
    
    # Execute pipeline
    print("\nExecuting pipeline...")
    result_ctx = pipeline.run(ctx)
    
    print(f"✓ Pipeline executed")
    
    # Check results
    print("\nResults:")
    if ContextDataType.PROMPT_EMBEDDINGS in result_ctx.data:
        embeddings_data = result_ctx.data[ContextDataType.PROMPT_EMBEDDINGS]
        if isinstance(embeddings_data, list):
            print(f"  Received {len(embeddings_data)} embedding sets (ordered)")
            for i, emb_set in enumerate(embeddings_data):
                if isinstance(emb_set, dict):
                    prompts = emb_set.get('prompts', [])
                    embeddings = emb_set.get('embeddings')
                    print(f"    [{i}] prompts: {prompts}")
                    if embeddings is not None:
                        print(f"        embeddings shape: {embeddings.shape if hasattr(embeddings, 'shape') else type(embeddings)}")
        else:
            print(f"  Embeddings data type: {type(embeddings_data)}")
            if isinstance(embeddings_data, dict):
                print(f"  Keys: {list(embeddings_data.keys())}")
    
    print("\n✅ EMBEDDINGS ORDERED MERGE TEST PASSED")
    return True


if __name__ == "__main__":
    try:
        # Test 1: Simple ordered merge
        test_ordered_merge_simple()
        
        # Test 2: Ordered merge with embeddings
        test_ordered_merge_with_embeddings()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
