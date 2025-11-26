#!/usr/bin/env python3
"""Test script to debug SmolVLM pipeline issue."""

import sys
import os
from pathlib import Path

# Set up paths like main.py does
PROJECT_ROOT = Path(__file__).parent / "src"
REPO_ROOT = Path(__file__).parent
for p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Set up logging to see DEBUG messages
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s:%(name)s:%(message)s'
)

from src.pipeline.dsl_parser import DSLParser, create_task_registry
from src.pipeline.task_base import Context, ContextDataType
from src.pipeline.pipeline_runner import PipelineRunner

def main():
    """Run the pipeline with test input."""
    # Parse DSL
    print("Parsing DSL...")
    with open("pipelines/smolvlm_chat.dsl") as f:
        dsl_text = f.read()
    
    # Get task registry
    task_registry = create_task_registry()
    print(f"Task registry has {len(task_registry)} tasks")
    
    parser = DSLParser(task_registry)
    pipeline = parser.parse(dsl_text)
    print(f"Pipeline parsed: {pipeline}")
    
    # Create runner
    print("Creating runner...")
    runner = PipelineRunner(pipeline)
    
    # Create context
    context = Context()
    
    # Start pipeline in background thread
    print("Starting pipeline...")
    import threading
    def run():
        try:
            runner.run(context)
        except Exception as e:
            print(f"Pipeline error: {e}")
            import traceback
            traceback.print_exc()
    
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    
    import time
    time.sleep(0.5)  # Let it start
    
    # Send input
    print("Sending input: 'describe the image'")
    runner.send_input("describe the image")
    
    # Wait for output
    print("Waiting for output...")
    for i in range(100):  # Wait up to 10 seconds
        output = runner.get_output(timeout=0.1)
        if output is not None:
            print(f"Got output: {output}")
            break
        if not thread.is_alive():
            print("Thread died")
            break
    else:
        print("Timeout waiting for output")
    
    # Send empty to exit
    print("Sending empty input to exit...")
    runner.send_input("")
    
    # Wait for thread
    thread.join(timeout=2)
    print("Done")

if __name__ == "__main__":
    main()
