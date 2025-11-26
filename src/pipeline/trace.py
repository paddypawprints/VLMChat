"""
Execution trace recording for pipeline tasks.

Provides pluggable trace backends for recording task execution:
- InMemoryTrace: Store events in context data (default)
- LogTrace: Emit events to logger (future)
- StreamTrace: Write events to file/stream (future)
- NoOpTrace: Disable tracing (future)
"""

import logging
import time
import threading
from typing import List, Tuple, Protocol
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# Type alias for trace events
# Format: (timestamp, thread_id, task_id, upstream_task_ids, event_type, submission_time, exit_code)
# exit_code: 0=success, non-zero=failure (unix convention)
TraceEvent = Tuple[float, int, str, List[str], str, float, int]


class BaseTrace(ABC):
    """
    Abstract base class for trace recording strategies.
    
    Implementations can store traces in memory, emit to logs,
    write to files, or send to external monitoring systems.
    """
    
    @abstractmethod
    def record_event(
        self,
        task_id: str,
        upstream_task_ids: List[str],
        event_type: str,
        submission_time: float = None,
        exit_code: int = 0
    ) -> None:
        """
        Record a trace event.
        
        Args:
            task_id: ID of the task being executed
            upstream_task_ids: List of upstream task IDs (dependencies)
            event_type: Type of event ('execute', 'split', 'merge', etc.)
            submission_time: When task was submitted to thread pool (None if not pooled)
            exit_code: Task exit code (0=success, non-zero=failure, unix convention)
        """
        pass
    
    @abstractmethod
    def get_events(self) -> List[TraceEvent]:
        """
        Retrieve all recorded trace events.
        
        Returns:
            List of trace events in chronological order
        """
        pass
    
    @abstractmethod
    def clear(self) -> None:
        """Clear all recorded events."""
        pass


class InMemoryTrace(BaseTrace):
    """
    Default trace implementation that stores events in memory.
    
    Events are stored as a list of tuples in chronological order.
    This implementation is used by default and stores traces in
    the Context.data[ContextDataType.TRACE] for portability across
    splits and merges.
    """
    
    def __init__(self):
        self.events: List[TraceEvent] = []
    
    def record_event(
        self,
        task_id: str,
        upstream_task_ids: List[str],
        event_type: str,
        submission_time: float = None,
        exit_code: int = 0
    ) -> None:
        """Record event with current timestamp, thread, and exit code."""
        event = (
            time.time(),
            threading.get_ident(),
            task_id,
            upstream_task_ids,
            event_type,
            submission_time if submission_time is not None else time.time(),
            exit_code
        )
        self.events.append(event)
    
    def get_events(self) -> List[TraceEvent]:
        """Return all events in chronological order."""
        return sorted(self.events, key=lambda e: e[0])
    
    def clear(self) -> None:
        """Clear all recorded events."""
        self.events.clear()


class LogTrace(BaseTrace):
    """
    Trace implementation that emits events to logger.
    
    Events are logged at INFO level with structured format.
    Useful for production environments with log aggregation.
    
    Note: Events are not stored in memory, only logged.
    get_events() returns empty list.
    """
    
    def __init__(self, logger_name: str = "pipeline.trace"):
        self.logger = logging.getLogger(logger_name)
    
    def record_event(
        self,
        task_id: str,
        upstream_task_ids: List[str],
        event_type: str,
        submission_time: float = None,
        exit_code: int = 0
    ) -> None:
        """Log event with structured format including exit code."""
        upstreams_str = ', '.join(upstream_task_ids) if upstream_task_ids else '-'
        exit_status = 'OK' if exit_code == 0 else f'FAIL({exit_code})'
        self.logger.info(
            f"TRACE: {event_type:8s} | task={task_id:20s} | "
            f"upstreams=[{upstreams_str}] | "
            f"thread={threading.get_ident()} | "
            f"exit={exit_status}"
        )
    
    def get_events(self) -> List[TraceEvent]:
        """
        Log trace does not store events.
        
        Returns empty list. Use log aggregation to retrieve events.
        """
        return []
    
    def clear(self) -> None:
        """No-op for log trace (events are not stored)."""
        pass


class NoOpTrace(BaseTrace):
    """
    Trace implementation that does nothing.
    
    Used to completely disable tracing for performance-critical
    pipelines where trace overhead is not acceptable.
    """
    
    def record_event(
        self,
        task_id: str,
        upstream_task_ids: List[str],
        event_type: str,
        submission_time: float = None,
        exit_code: int = 0
    ) -> None:
        """No-op: discard event."""
        pass
    
    def get_events(self) -> List[TraceEvent]:
        """Return empty list."""
        return []
    
    def clear(self) -> None:
        """No-op."""
        pass


def print_trace_events(events: List[TraceEvent]) -> None:
    """
    Print formatted execution trace.
    
    Shows chronological execution with thread IDs and dependencies.
    
    Args:
        events: List of trace events (timestamp, thread_id, task_id, upstream_ids, event_type, submission_time)
    """
    if not events:
        print("No trace events")
        return
    
    print("\n" + "="*70)
    print("EXECUTION TRACE")
    print("="*70)
    print(f"{'Time':>12} | {'Wait':>8} | {'Thread':>12} | {'Task':<20} | {'Upstreams':<20} | {'Event':<10} | {'Exit'}")
    print("-"*70)
    
    # Sort by timestamp
    sorted_events = sorted(events, key=lambda x: x[0])
    start_time = sorted_events[0][0]
    
    for event in sorted_events:
        # New format: 7-tuple with submission_time and exit_code
        timestamp, thread_id, task_id, upstream_ids, event_type, submission_time, exit_code = event
        wait_ms = (timestamp - submission_time) * 1000 if submission_time else None
        
        elapsed_ms = (timestamp - start_time) * 1000
        upstreams_str = ", ".join(upstream_ids) if upstream_ids else "-"
        wait_str = f"{wait_ms:.1f}ms" if wait_ms is not None else "-"
        exit_str = "OK" if exit_code == 0 else f"FAIL({exit_code})"
        print(f"{elapsed_ms:>11.1f}ms | {wait_str:>8} | {thread_id:>12} | {task_id:<20} | {upstreams_str:<20} | {event_type:<10} | {exit_str}")
    
    print("="*70)
    print(f"Total events: {len(events)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    """Test trace implementations."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(message)s'
    )
    
    print("="*70)
    print("Trace Implementation Tests")
    print("="*70)
    
    # Test 1: InMemoryTrace
    print("\n1. InMemoryTrace:")
    memory_trace = InMemoryTrace()
    memory_trace.record_event("task1", [], "execute", exit_code=0)
    time.sleep(0.01)
    memory_trace.record_event("task2", ["task1"], "execute", exit_code=0)
    time.sleep(0.01)
    memory_trace.record_event("fork", ["task2"], "split", exit_code=0)
    
    events = memory_trace.get_events()
    assert len(events) == 3
    print(f"✅ Recorded {len(events)} events")
    print_trace_events(events)
    
    # Test 2: LogTrace
    print("\n2. LogTrace (check console output):")
    log_trace = LogTrace()
    log_trace.record_event("task_a", [], "execute", exit_code=0)
    log_trace.record_event("task_b", ["task_a"], "execute", exit_code=1)  # Simulate failure
    log_trace.record_event("merge", ["task_a", "task_b"], "merge", exit_code=0)
    print("✅ Events logged (see above)")
    
    # Test 3: NoOpTrace
    print("\n3. NoOpTrace:")
    noop_trace = NoOpTrace()
    noop_trace.record_event("task1", [], "execute", exit_code=0)
    noop_trace.record_event("task2", ["task1"], "execute", exit_code=0)
    events = noop_trace.get_events()
    assert len(events) == 0
    print("✅ No events recorded (as expected)")
    
    # Test 4: Clear
    print("\n4. Clear events:")
    memory_trace.clear()
    events = memory_trace.get_events()
    assert len(events) == 0
    print("✅ Events cleared")
    
    print("\n" + "="*70)
    print("All trace tests passed!")
    print("="*70)
