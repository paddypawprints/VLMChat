"""
Minimal test utilities for smoke tests.
Each test is standalone and can run independently.
"""

import threading
import time


def run_dsl_pipeline(dsl_text: str, max_workers: int = 2, timeout_seconds: float = 10.0):
    """
    Parse and execute DSL pipeline with timeout protection, return result context.
    
    Args:
        dsl_text: DSL pipeline text to parse and execute
        max_workers: Number of worker threads for pipeline runner
        timeout_seconds: Maximum execution time before raising TimeoutError
        
    Returns:
        Context object with execution results
        
    Raises:
        TimeoutError: If pipeline execution exceeds timeout_seconds
    """
    from vlmchat.pipeline.dsl_parser import DSLParser, create_task_registry
    from vlmchat.pipeline.pipeline_runner import PipelineRunner
    from vlmchat.pipeline.task_base import Context
    
    registry = create_task_registry()
    parser = DSLParser(registry)
    
    # Parser returns (sources_dict, ast_root)
    sources, ast_root = parser.parse(dsl_text)
    
    runner = PipelineRunner(ast_root=ast_root, max_workers=max_workers)
    
    # Register sources if any
    if sources:
        for name, source_config in sources.items():
            runner.sources[name] = source_config
    
    # Run with timeout protection
    context = Context()
    result_container: list = [None]
    exception_container: list = [None]
    
    def run_with_exception_capture():
        try:
            result_container[0] = runner.run(context)
        except Exception as e:
            exception_container[0] = e
    
    thread = threading.Thread(target=run_with_exception_capture, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Thread still running - timeout occurred
        runner.shutdown()
        raise TimeoutError(
            f"Pipeline execution exceeded {timeout_seconds}s timeout. "
            f"Possible infinite loop in DSL:\n{dsl_text[:200]}"
        )
    
    runner.shutdown()
    
    # Re-raise any exception from execution thread
    if exception_container[0]:
        raise exception_container[0]
    
    return result_container[0]


def get_counter_value(context, counter_name: str) -> int:
    """
    Extract diagnostic counter value from context.
    
    Args:
        context: Pipeline context with DIAGNOSTIC data
        counter_name: Name of counter to extract
        
    Returns:
        Counter value, or 0 if not found
    """
    from vlmchat.pipeline.task_base import ContextDataType
    
    if ContextDataType.DIAGNOSTIC not in context.data:
        return 0
    
    counter_key = f"_counter_{counter_name}"
    for item in context.data[ContextDataType.DIAGNOSTIC]:
        if isinstance(item, dict) and item.get('key') == counter_key:
            return item.get('value', 0)
    return 0


# ============================================================================
# Legacy helpers (kept for backwards compatibility with existing tests)
# ============================================================================

from typing import List, Optional


def get_executed_tasks(trace_events) -> List[str]:
    """
    Extract list of executed task IDs from trace events.
    
    Args:
        trace_events: List of trace events from PipelineRunner
        
    Returns:
        List of task IDs that were executed
    """
    return [event[2] for event in trace_events if len(event) >= 5 and event[4] == 'execute']


def get_task_count(trace_events, task_name: str) -> int:
    """
    Count how many times a task was executed.
    
    Args:
        trace_events: List of trace events from PipelineRunner
        task_name: Name or ID of the task to count
        
    Returns:
        Number of times the task was executed
    """
    executed = get_executed_tasks(trace_events)
    return sum(1 for task_id in executed if task_name in task_id)


def assert_task_executed(trace_events, task_name: str, message: Optional[str] = None):
    """
    Assert that a task was executed at least once.
    
    Args:
        trace_events: List of trace events from PipelineRunner
        task_name: Name or ID of the task
        message: Optional custom error message
    """
    executed = get_executed_tasks(trace_events)
    assert any(task_name in task_id for task_id in executed), \
        message or f"Expected task '{task_name}' to be executed. Executed: {executed}"


def assert_task_not_executed(trace_events, task_name: str, message: Optional[str] = None):
    """
    Assert that a task was NOT executed.
    
    Args:
        trace_events: List of trace events from PipelineRunner
        task_name: Name or ID of the task
        message: Optional custom error message
    """
    executed = get_executed_tasks(trace_events)
    assert not any(task_name in task_id for task_id in executed), \
        message or f"Expected task '{task_name}' NOT to be executed. Executed: {executed}"


def assert_execution_order(trace_events, *task_names):
    """
    Assert that tasks were executed in the specified order.
    
    Args:
        trace_events: List of trace events from PipelineRunner
        *task_names: Task names in expected order
    """
    executed = get_executed_tasks(trace_events)
    indices = []
    
    for task_name in task_names:
        # Find first occurrence of this task
        idx = next((i for i, task_id in enumerate(executed) if task_name in task_id), None)
        assert idx is not None, f"Task '{task_name}' not found in executed tasks: {executed}"
        indices.append(idx)
    
    # Verify order
    assert indices == sorted(indices), \
        f"Tasks not in expected order. Expected: {task_names}, Executed: {executed}"
