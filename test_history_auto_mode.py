#!/usr/bin/env python3
"""Test history_update auto mode detection."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.pipeline.tasks.history_update_task import HistoryUpdateTask
from src.pipeline.task_base import Context, ContextDataType

def test_history_auto_mode():
    """Test automatic mode detection in history_update."""
    
    # Create task instance
    task = HistoryUpdateTask(task_id="hist")
    task.configure(format="simple")
    
    # Create context
    context = Context()
    
    print("=== Test 1: First call (should be PROMPT mode) ===")
    context.data[ContextDataType.TEXT] = ["What is 2+2?"]
    context = task.run(context)
    print(f"TEXT after first call: {context.data[ContextDataType.TEXT]}")
    print(f"History pairs: {task.history._pairs}")
    assert len(context.data[ContextDataType.TEXT]) == 2, "Should have 2 TEXT items"
    assert context.data[ContextDataType.TEXT][0] == "", "First should be empty history"
    assert context.data[ContextDataType.TEXT][1] == "What is 2+2?", "Second should be user input"
    print("✅ PROMPT mode detected correctly\n")
    
    print("=== Test 2: Second call (should be RESPONSE mode) ===")
    context.data[ContextDataType.TEXT].append("The answer is 4")
    context = task.run(context)
    print(f"TEXT after second call: {context.data[ContextDataType.TEXT]}")
    print(f"History pairs: {task.history._pairs}")
    assert task.history._pairs[0] == ("What is 2+2?", "The answer is 4"), "History should have complete pair"
    print("✅ RESPONSE mode detected correctly\n")
    
    print("=== Test 3: Third call (should be PROMPT mode again) ===")
    # Clear TEXT for new iteration
    context.data[ContextDataType.TEXT] = ["What is 3+3?"]
    context = task.run(context)
    print(f"TEXT after third call: {context.data[ContextDataType.TEXT]}")
    print(f"History pairs: {task.history._pairs}")
    assert len(context.data[ContextDataType.TEXT]) == 2, "Should have 2 TEXT items"
    # History should now have first exchange
    assert "What is 2+2?" in context.data[ContextDataType.TEXT][0], "History should include first exchange"
    assert context.data[ContextDataType.TEXT][1] == "What is 3+3?", "Second should be new user input"
    print("✅ PROMPT mode detected correctly (with history)\n")
    
    print("=== Test 4: Fourth call (should be RESPONSE mode again) ===")
    context.data[ContextDataType.TEXT].append("The answer is 6")
    context = task.run(context)
    print(f"TEXT after fourth call: {context.data[ContextDataType.TEXT]}")
    print(f"History pairs: {task.history._pairs}")
    assert len(task.history._pairs) == 2, "Should have 2 pairs in history"
    assert task.history._pairs[1] == ("What is 3+3?", "The answer is 6"), "Second pair should be complete"
    print("✅ RESPONSE mode detected correctly\n")
    
    print("="*50)
    print("✅ All tests passed! Auto mode detection works perfectly.")
    print(f"Final history:\n{task.history.get_formatted_history()}")

if __name__ == "__main__":
    test_history_auto_mode()
