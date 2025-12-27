#!/usr/bin/env python3
"""
Test SmolVLM pipeline with history and camera.

Tests the complete pipeline:
camera + console_input -> history(prompt) -> smolvlm -> history(response) -> console_output
"""

import sys
import os
from pathlib import Path
import logging

# Ensure we're using the correct Python path for local imports
PROJECT_ROOT = Path(__file__).parent / "src"
REPO_ROOT = Path(__file__).parent
for p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.pipeline.dsl_parser import DSLParser, create_task_registry
from src.pipeline.pipeline_runner import PipelineRunner
from src.pipeline.task_base import Context

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

def test_smolvlm_chat():
    """Test SmolVLM chat pipeline with history."""
    print("\n" + "="*60)
    print("TEST: SmolVLM Chat Pipeline")
    print("="*60)
    
    # Load pipeline from DSL
    registry = create_task_registry()
    parser = DSLParser(registry, pipeline_dirs=['./pipelines'])
    
    dsl_file = './pipelines/smolvlm_chat.dsl'
    
    print(f"\nLoading pipeline from: {dsl_file}")
    
    with open(dsl_file, 'r') as f:
        dsl_text = f.read()
    
    print(f"DSL: {dsl_text}\n")
    
    pipeline = parser.parse(dsl_text)
    
    # Pipeline can be a single task (loop) or list of tasks
    if isinstance(pipeline, list):
        print(f"✓ Pipeline parsed: {len(pipeline)} tasks")
        for i, task in enumerate(pipeline):
            print(f"  Task {i}: {task.__class__.__name__} (id={task.task_id})")
    else:
        print(f"✓ Pipeline parsed: {pipeline.__class__.__name__} (id={pipeline.task_id})")
    
    # Create runner with the pipeline
    runner = PipelineRunner(pipeline)
    context = Context()
    
    print("\n" + "="*60)
    print("Starting interactive chat (empty input to exit)...")
    print("="*60 + "\n")
    
    # Run the pipeline
    try:
        result = runner.run(context)
        print("\n✓ Pipeline completed successfully")
        
        # Show final context state
        from src.pipeline.task_base import ContextDataType
        if ContextDataType.TEXT in result.data:
            print(f"\nFinal TEXT entries: {len(result.data[ContextDataType.TEXT])}")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    test_smolvlm_chat()
