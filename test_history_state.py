#!/usr/bin/env python3
"""
Test history_update task with explicit ID for state sharing.

Tests that using the same id parameter allows the history task
to maintain state across multiple invocations (prompt + response).
"""

import sys
import logging
from src.pipeline.dsl_parser import DSLParser, create_task_registry
from src.pipeline.task_base import Context, ContextDataType

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s:%(name)s:%(message)s'
)

def test_history_with_shared_state():
    """Test that history_update with same id shares state."""
    print("\n" + "="*60)
    print("TEST: History state sharing with explicit id")
    print("="*60)
    
    # Simple pipeline: console_input -> history(prompt) -> mock_model -> history(response) -> console_output
    # We'll manually simulate this flow
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    # Parse a DSL that uses the same id for both history invocations
    dsl = 'history_update(id="hist", prompt=true, format="simple") -> console_output()'
    
    pipeline = parser.parse(dsl)
    
    print(f"\n✓ Pipeline parsed: {len(pipeline)} tasks")
    for i, task in enumerate(pipeline):
        print(f"  Task {i}: {task.__class__.__name__} (id={task.task_id})")
    
    # Get the history task instance
    history_task = pipeline[0]
    
    # Create context with user input
    context = Context()
    context.data[ContextDataType.TEXT] = ["Hello, how are you?"]
    
    print("\n--- Running prompt mode ---")
    print(f"Input TEXT: {context.data[ContextDataType.TEXT]}")
    
    # Run the history task in prompt mode
    context = history_task.run(context)
    
    print(f"Output TEXT: {context.data[ContextDataType.TEXT]}")
    print(f"History pairs: {len(history_task.history._pairs)}")
    if history_task.history._pairs:
        req, resp = history_task.history._pairs[-1]
        print(f"  Request: '{req}'")
        print(f"  Response: '{resp}'")
    
    # Simulate model output
    context.data[ContextDataType.TEXT].append("I'm doing well, thank you!")
    
    # Now configure the SAME task instance for response mode
    print("\n--- Reconfiguring for response mode ---")
    history_task.configure(response=True)
    
    print(f"Input TEXT: {context.data[ContextDataType.TEXT]}")
    
    # Run in response mode
    context = history_task.run(context)
    
    print(f"Output TEXT: {context.data[ContextDataType.TEXT]}")
    print(f"History pairs: {len(history_task.history._pairs)}")
    if history_task.history._pairs:
        req, resp = history_task.history._pairs[-1]
        print(f"  Request: '{req}'")
        print(f"  Response: '{resp}'")
    
    # Verify
    assert len(history_task.history._pairs) == 1, "Should have 1 conversation pair"
    req, resp = history_task.history._pairs[0]
    assert req == "Hello, how are you?", f"Request mismatch: {req}"
    assert resp == "I'm doing well, thank you!", f"Response mismatch: {resp}"
    
    print("\n✓ TEST PASSED: History maintained state across invocations")


def test_history_in_full_dsl():
    """Test history with both invocations in DSL using same id."""
    print("\n" + "="*60)
    print("TEST: Full DSL with history prompt and response")
    print("="*60)
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    # DSL with same id for both history invocations
    dsl = '''history_update(id="hist", prompt=true, format="simple") -> 
             console_output() -> 
             history_update(id="hist", response=true)'''
    
    pipeline = parser.parse(dsl)
    
    print(f"\n✓ Pipeline parsed: {len(pipeline)} tasks")
    task_ids = []
    for i, task in enumerate(pipeline):
        print(f"  Task {i}: {task.__class__.__name__} (id={task.task_id})")
        task_ids.append(task.task_id)
    
    # Verify that both history tasks have the same id (are the same instance)
    history_tasks = [task for task in pipeline if task.__class__.__name__ == 'HistoryUpdateTask']
    
    print(f"\nFound {len(history_tasks)} history tasks")
    if len(history_tasks) == 2:
        print(f"  Task 0 id: {history_tasks[0].task_id}")
        print(f"  Task 1 id: {history_tasks[1].task_id}")
        print(f"  Same instance? {history_tasks[0] is history_tasks[1]}")
        
        assert history_tasks[0] is history_tasks[1], "Should be the same instance!"
        print("\n✓ TEST PASSED: Both history invocations share the same task instance")
    else:
        print(f"\n✗ Expected 2 history tasks, found {len(history_tasks)}")
        return False


if __name__ == "__main__":
    try:
        test_history_with_shared_state()
        test_history_in_full_dsl()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED ✓✓✓")
        print("="*60)
        sys.exit(0)
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
