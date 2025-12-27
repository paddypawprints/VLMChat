#!/usr/bin/env python3
"""
Test building CLIP detection pipeline from DSL.

Tests that the DSL can be parsed and built into an executable pipeline.
"""

import sys
from pathlib import Path

# Set up Python paths correctly for package imports
# Tasks use relative imports like "from ...camera" which need proper package structure
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"

# Add both repo root (for "src.camera" imports) and src dir (for "camera" imports)
for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.config import VLMChatConfig
from src.pipeline.dsl_parser import DSLParser, create_task_registry
from src.metrics.metrics_collector import Collector


def test_clip_pipeline_build():
    """Test building clip_detection.dsl into a pipeline."""
    print("\n" + "="*70)
    print("TEST: Build CLIP Detection Pipeline from DSL")
    print("="*70)
    
    # Load configuration
    config = VLMChatConfig()
    collector = Collector()
    
    # Get task registry
    print("\nLoading task registry...")
    registry = create_task_registry()
    print(f"  ✓ Registry loaded with {len(registry)} tasks")
    
    # Check for required tasks
    required_tasks = [
        "input", "camera", "detector", "filter",
        "clip_text_encoder", "clip_vision", "clip_comparator", "output"
    ]
    
    missing_tasks = [task for task in required_tasks if task not in registry]
    if missing_tasks:
        print(f"\n⚠️  Missing tasks in registry: {missing_tasks}")
    else:
        print(f"  ✓ All required tasks present in registry")
    
    # Read DSL file
    dsl_path = Path(__file__).parent / "pipelines" / "clip_detection.dsl"
    with open(dsl_path, 'r') as f:
        dsl_text = f.read()
    
    print(f"\nParsing DSL: {dsl_path}")
    
    # Create parser
    parser = DSLParser(task_registry=registry)
    
    try:
        # Parse and build pipeline
        print("  Building pipeline...")
        pipeline = parser.parse(dsl_text)
        print(f"  ✓ Pipeline built successfully")
        print(f"    Type: {type(pipeline).__name__}")
        
        # Check pipeline structure
        if hasattr(pipeline, 'tasks'):
            print(f"    Tasks: {len(pipeline.tasks)}")
            for i, task in enumerate(pipeline.tasks[:5]):  # First 5 tasks
                task_name = getattr(task, 'task_id', type(task).__name__)
                print(f"      [{i}] {task_name}")
        
        print("\n✅ PIPELINE BUILD SUCCESSFUL")
        return True
        
    except Exception as e:
        print(f"\n❌ PIPELINE BUILD FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    try:
        success = test_clip_pipeline_build()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
