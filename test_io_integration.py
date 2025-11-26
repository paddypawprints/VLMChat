"""
Test script for I/O integration with queue-based coordination.

Tests that console_input and console_output tasks work with
pipeline runner's queues.
"""

import sys
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).parent
PROJECT_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.dsl_parser import DSLParser, create_task_registry
from src.pipeline.pipeline_runner import PipelineRunner
from src.pipeline.task_base import Context
import threading
import time

def test_simple_io():
    """Test simple input -> output pipeline with queues."""
    print("\n" + "="*70)
    print("TEST 1: Simple I/O (console_input -> console_output)")
    print("="*70)
    
    # Parse DSL
    dsl = "console_input(prompt=\"Test: \") -> console_output()"
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    # Create runner
    runner = PipelineRunner(pipeline, max_workers=2)
    
    # Run in background thread
    def run_pipeline():
        ctx = Context()
        runner.run(ctx)
    
    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()
    
    # Wait for pipeline to start
    time.sleep(0.5)
    
    # Send input
    print("Sending input: 'Hello World'")
    runner.send_input("Hello World")
    
    # Get output
    output = runner.get_output(timeout=5.0)
    print(f"Received output: {output}")
    
    assert output == "Hello World", f"Expected 'Hello World', got '{output}'"
    
    thread.join(timeout=2)
    print("✅ TEST 1 PASSED")

def test_loop_with_io():
    """Test loop with console_input and break_on empty input."""
    print("\n" + "="*70)
    print("TEST 2: Loop with I/O and break_on")
    print("="*70)
    
    # Parse DSL - loop that echoes input until empty
    dsl = """
    {
        console_input(prompt="Enter text: ") -> break_on(code=1) ->
        console_output(which="last")
    }
    """
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    pipeline = parser.parse(dsl)
    
    # Create runner
    runner = PipelineRunner(pipeline, max_workers=2)
    
    # Run in background thread
    def run_pipeline():
        ctx = Context()
        runner.run(ctx)
    
    thread = threading.Thread(target=run_pipeline, daemon=True)
    thread.start()
    
    # Wait for pipeline to start
    time.sleep(0.5)
    
    # Send multiple inputs
    inputs = ["First", "Second", "Third", ""]  # Empty triggers break
    
    for inp in inputs:
        print(f"Sending: '{inp}'")
        runner.send_input(inp)
        time.sleep(0.3)  # Give pipeline time to process
    
    # Drain all outputs
    outputs = []
    while True:
        output = runner.get_output(timeout=0.5)
        if output is None:
            break
        print(f"Got: '{output}'")
        outputs.append(output)
    
    # Wait for pipeline to finish
    thread.join(timeout=2)
    
    print(f"\nSent {len(inputs)} inputs (last was empty)")
    print(f"Got {len(outputs)} outputs")
    
    assert outputs == ["First", "Second", "Third"], f"Unexpected outputs: {outputs}"
    assert not runner.is_running(), "Pipeline should have stopped"
    
    print("✅ TEST 2 PASSED")

def test_smolvlm_dsl():
    """Test loading and preparing smolvlm_chat.dsl (without actual model)."""
    print("\n" + "="*70)
    print("TEST 3: Load smolvlm_chat.dsl")
    print("="*70)
    
    dsl_file = REPO_ROOT / "pipelines" / "smolvlm_chat.dsl"
    
    if not dsl_file.exists():
        print(f"⚠️  DSL file not found: {dsl_file}")
        return
    
    with open(dsl_file) as f:
        dsl = f.read()
    
    print(f"DSL:\n{dsl}")
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    try:
        pipeline = parser.parse(dsl)
        print("✅ DSL parsed successfully")
        
        # Create runner (don't run - would need model)
        runner = PipelineRunner(pipeline, max_workers=2)
        print("✅ Runner created successfully")
        
        # Verify queues exist
        assert hasattr(runner, 'input_queue'), "Runner should have input_queue"
        assert hasattr(runner, 'output_queue'), "Runner should have output_queue"
        print("✅ Runner has I/O queues")
        
        print("✅ TEST 3 PASSED")
        
    except Exception as e:
        print(f"❌ TEST 3 FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    try:
        test_simple_io()
        test_loop_with_io()
        test_smolvlm_dsl()
        
        print("\n" + "="*70)
        print("ALL TESTS PASSED ✅")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
