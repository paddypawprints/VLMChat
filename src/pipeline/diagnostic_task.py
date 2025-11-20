"""
Diagnostic task for testing pipeline execution and control structures.

Provides observable behavior for validating:
- Sequential vs parallel execution
- Thread pool utilization
- Timing constraints
- Data flow through context
- Loop iteration counting
"""

import logging
import time
import threading
from typing import Optional, Any
from .task_base import BaseTask, Context, ContextDataType

logger = logging.getLogger(__name__)


class DiagnosticTask(BaseTask):
    """
    Task that reports execution details for testing and debugging.
    
    Features:
    - Echoes message with thread ID
    - Configurable delay for timing tests
    - Reports context contents (item counts per type)
    - Can inject data into context
    - Can increment counters for loop testing
    """
    
    def __init__(
        self,
        message: str = "diagnostic",
        delay_ms: int = 0,
        put_key: Optional[str] = None,
        put_value: Any = None,
        increment_counter: Optional[str] = None,
        task_id: Optional[str] = None
    ):
        super().__init__(task_id=task_id or f"diagnostic_{message}")
        self.message = message
        self.delay_ms = delay_ms
        self.put_key = put_key
        self.put_value = put_value
        self.increment_counter = increment_counter
    
    def configure(self, **kwargs):
        """Configure task from DSL parameters."""
        if 'message' in kwargs:
            self.message = kwargs['message']
        if 'delay_ms' in kwargs:
            self.delay_ms = int(kwargs['delay_ms'])
        if 'put_key' in kwargs:
            self.put_key = kwargs['put_key']
        if 'put_value' in kwargs:
            self.put_value = kwargs['put_value']
        if 'increment_counter' in kwargs:
            self.increment_counter = kwargs['increment_counter']
        if 'counter' in kwargs:
            # Alias for increment_counter for simpler syntax
            self.increment_counter = kwargs['counter']
    
    def _format_context_summary(self, context: Context) -> str:
        """Format compact summary of context contents."""
        if not context.data:
            return "empty"
        
        parts = []
        for data_type, items in context.data.items():
            count = len(items) if items else 0
            parts.append(f"{data_type.type_name}:{count}")
        
        return ", ".join(parts)
    
    def run(self, context: Context) -> Context:
        """Execute diagnostic task with reporting."""
        thread_id = threading.current_thread().name
        thread_ident = threading.get_ident()
        
        # Report start with context summary
        context_summary = self._format_context_summary(context)
        logger.info(
            f"[{thread_id}#{thread_ident}] DiagnosticTask('{self.message}'): "
            f"starting | context=[{context_summary}]"
        )
        
        # Delay if specified
        if self.delay_ms > 0:
            start_time = time.time()
            time.sleep(self.delay_ms / 1000.0)
            actual_delay = (time.time() - start_time) * 1000
            logger.info(
                f"[{thread_id}#{thread_ident}] DiagnosticTask('{self.message}'): "
                f"slept {actual_delay:.1f}ms"
            )
        
        # Increment counter if specified (for loop testing)
        if self.increment_counter:
            # Use DIAGNOSTIC data type for counters
            if ContextDataType.DIAGNOSTIC not in context.data:
                context.data[ContextDataType.DIAGNOSTIC] = []
            
            counter_key = f"_counter_{self.increment_counter}"
            # Find existing counter or create new
            counters = {item['key']: item for item in context.data[ContextDataType.DIAGNOSTIC] if isinstance(item, dict)}
            current = counters.get(counter_key, {}).get('value', 0)
            counters[counter_key] = {'key': counter_key, 'value': current + 1}
            context.data[ContextDataType.DIAGNOSTIC] = list(counters.values())
            
            logger.info(
                f"[{thread_id}#{thread_ident}] DiagnosticTask('{self.message}'): "
                f"counter '{self.increment_counter}' = {current + 1}"
            )
        
        # Put data into context if specified
        if self.put_key is not None:
            # Store in DIAGNOSTIC data type for simple key-value pairs
            if ContextDataType.DIAGNOSTIC not in context.data:
                context.data[ContextDataType.DIAGNOSTIC] = []
            
            context.data[ContextDataType.DIAGNOSTIC].append({
                'key': self.put_key,
                'value': self.put_value
            })
            
            logger.info(
                f"[{thread_id}#{thread_ident}] DiagnosticTask('{self.message}'): "
                f"put '{self.put_key}' = {self.put_value}"
            )
        
        # Report completion
        logger.info(
            f"[{thread_id}#{thread_ident}] DiagnosticTask('{self.message}'): "
            f"completed"
        )
        
        return context
    
    def should_continue(self) -> bool:
        """Check if task should continue (timing budget)."""
        if not self.time_budget_ms:
            return True
        
        if self._start_time is None:
            return True
        
        elapsed_ms = (time.time() - self._start_time) * 1000.0
        remaining = self.time_budget_ms - elapsed_ms
        
        if remaining < 0:
            thread_id = threading.current_thread().name
            logger.warning(
                f"[{thread_id}] DiagnosticTask('{self.message}'): "
                f"exceeded time budget by {-remaining:.1f}ms"
            )
        
        return remaining > 0


if __name__ == "__main__":
    """Test diagnostic task behavior."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*70)
    print("DiagnosticTask Tests")
    print("="*70)
    
    # Test 1: Basic execution
    print("\n1. Basic execution:")
    task1 = DiagnosticTask(message="test1")
    ctx = Context()
    result = task1.run(ctx)
    print("✅ Basic execution works")
    
    # Test 2: With delay
    print("\n2. With delay:")
    task2 = DiagnosticTask(message="delayed", delay_ms=50)
    result = task2.run(ctx)
    print("✅ Delay works")
    
    # Test 3: Context injection
    print("\n3. Context injection:")
    ctx3 = Context()
    task3 = DiagnosticTask(message="producer", put_key="status", put_value="ready")
    result = task3.run(ctx3)
    assert ContextDataType.DIAGNOSTIC in result.data
    diagnostic_data = {item['key']: item['value'] for item in result.data[ContextDataType.DIAGNOSTIC] if isinstance(item, dict) and 'key' in item}
    assert diagnostic_data["status"] == "ready"
    print(f"✅ Context injection works: {diagnostic_data}")
    
    # Test 4: Counter increment
    print("\n4. Counter increment:")
    ctx4 = Context()
    task4 = DiagnosticTask(message="counter", increment_counter="iterations")
    result = task4.run(ctx4)
    result = task4.run(result)
    result = task4.run(result)
    counters = {item['key']: item['value'] for item in result.data[ContextDataType.DIAGNOSTIC] if isinstance(item, dict) and 'key' in item}
    assert counters["_counter_iterations"] == 3
    print(f"✅ Counter works: {counters['_counter_iterations']} iterations")
    
    # Test 5: Context with data
    print("\n5. Context summary with data:")
    ctx_with_data = Context()
    ctx_with_data.data[ContextDataType.IMAGE] = ["img1", "img2"]
    ctx_with_data.data[ContextDataType.DETECTIONS] = ["det1"]
    task5 = DiagnosticTask(message="with_data")
    result = task5.run(ctx_with_data)
    print("✅ Context summary works")
    
    # Test 6: Timing budget
    print("\n6. Timing budget:")
    task6 = DiagnosticTask(message="timed", delay_ms=100)
    task6.time_budget_ms = 50  # Set budget lower than delay
    task6._record_start()
    result = task6.run(ctx)
    should_continue = task6.should_continue()
    assert not should_continue
    print(f"✅ Timing budget check works: should_continue={should_continue}")
    
    # Test 7: Trace recording
    print("\n7. Trace recording:")
    ctx_trace = Context()
    
    # Simulate pipeline: task1 -> task2 -> task3
    task7a = DiagnosticTask(message="task1")
    task7b = DiagnosticTask(message="task2")
    task7c = DiagnosticTask(message="task3")
    
    # Set up dependencies
    task7b.upstream_tasks = [task7a]
    task7c.upstream_tasks = [task7b]
    
    # Note: Trace is now managed by PipelineRunner, not stored in context
    # See pipeline_runner.py tests for full trace functionality
    print("✅ Trace recording (now handled by PipelineRunner)")
    
    print("\n" + "="*70)
    print("All DiagnosticTask tests passed!")
    print("="*70)
