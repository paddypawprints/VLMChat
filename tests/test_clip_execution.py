#!/usr/bin/env python3
"""
Test executing CLIP detection pipeline with pre-configured prompts.
"""

import sys
from pathlib import Path
from io import StringIO

PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"

for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from src.utils.config import VLMChatConfig
from src.pipeline.dsl_parser import DSLParser, create_task_registry
from src.main.chat_services import VLMChatServices
from src.metrics.metrics_collector import Collector


def test_clip_execution():
    """Test executing clip_detection.dsl with pre-configured prompts."""
    print("\n" + "="*70)
    print("TEST: Execute CLIP Detection Pipeline")
    print("="*70)
    
    # Simulate user input with pre-configured prompts
    test_prompts = [
        "person",
        "horse",
        "bicycle",
        ""  # Empty to trigger break_on
    ]
    
    # Redirect stdin
    original_stdin = sys.stdin
    sys.stdin = StringIO("\n".join(test_prompts))
    
    try:
        config = VLMChatConfig()
        collector = Collector()
        
        # Load and parse DSL
        dsl_path = PROJECT_ROOT / "pipelines" / "clip_detection.dsl"
        with open(dsl_path) as f:
            dsl = f.read()
        
        print("\nParsing DSL...")
        registry = create_task_registry()
        parser = DSLParser(task_registry=registry)
        pipeline = parser.parse(dsl)
        
        print(f"Pipeline has {len(pipeline)} tasks/connectors:")
        for i, task in enumerate(pipeline):
            print(f"  [{i}] {type(task).__name__}")
        
        print("\n" + "-"*70)
        print("Executing pipeline...")
        print("-"*70 + "\n")
        
        # Execute pipeline with PipelineRunner
        from src.pipeline.pipeline_runner import PipelineRunner
        from src.pipeline.task_base import Context
        
        context = Context()
        context.config = config
        context.collector = collector
        
        runner = PipelineRunner(
            pipeline,
            collector=collector,
            enable_trace=True
        )
        
        result = runner.run(context)
        
        print("\n" + "-"*70)
        print("Pipeline execution complete!")
        print(f"Result type: {type(result)}")
        if result:
            print(f"Result: {result}")
        print("-"*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        sys.stdin = original_stdin


if __name__ == "__main__":
    try:
        success = test_clip_execution()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
