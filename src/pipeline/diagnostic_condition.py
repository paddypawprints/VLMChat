"""
Diagnostic condition for testing loop control flow.

Provides configurable loop exit conditions:
- Max iterations
- Timeout duration
- Custom predicate based on context data
"""

import logging
import time
import threading
from typing import Optional, Callable
from .task_base import LoopCondition, LoopControlAction, Context, ContextDataType

logger = logging.getLogger(__name__)


class DiagnosticCondition(LoopCondition):
    """
    Control task for testing loop behavior.
    
    Exit conditions (first to trigger wins):
    - max_iterations: Stop after N iterations
    - timeout_seconds: Stop after T seconds
    - predicate: Custom function(context) -> bool
    """
    
    def __init__(
        self,
        max_iterations: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        predicate: Optional[Callable[[Context], bool]] = None,
        task_id: Optional[str] = None
    ):
        super().__init__(task_id=task_id or "diagnostic_condition")
        self.max_iterations = max_iterations
        self.timeout_seconds = timeout_seconds
        self.predicate = predicate
        self.start_time = None
        self.iteration_count = 0
    
    def configure(self, **kwargs):
        """Configure condition from DSL parameters."""
        if 'max_iterations' in kwargs:
            self.max_iterations = int(kwargs['max_iterations'])
        if 'timeout_seconds' in kwargs:
            self.timeout_seconds = float(kwargs['timeout_seconds'])
        if 'seconds' in kwargs:
            self.timeout_seconds = float(kwargs['seconds'])
    
    def evaluate(self, context: Context) -> LoopControlAction:
        """
        Check exit conditions and return control action.
        
        Returns BREAK if loop should exit, PASS to continue.
        """
        thread_id = threading.current_thread().name
        thread_ident = threading.get_ident()
        
        # Initialize start time on first run
        if self.start_time is None:
            self.start_time = time.time()
            logger.info(
                f"[{thread_id}#{thread_ident}] DiagnosticCondition: "
                f"initialized (max_iter={self.max_iterations}, timeout={self.timeout_seconds}s)"
            )
        
        self.iteration_count += 1
        
        # Check max iterations
        if self.max_iterations is not None and self.iteration_count >= self.max_iterations:
            logger.info(
                f"[{thread_id}#{thread_ident}] DiagnosticCondition: "
                f"max iterations reached ({self.iteration_count}/{self.max_iterations})"
            )
            return LoopControlAction.BREAK
        
        # Check timeout
        if self.timeout_seconds is not None:
            elapsed = time.time() - self.start_time
            if elapsed >= self.timeout_seconds:
                logger.info(
                    f"[{thread_id}#{thread_ident}] DiagnosticCondition: "
                    f"timeout reached ({elapsed:.3f}s >= {self.timeout_seconds}s)"
                )
                return LoopControlAction.BREAK
        
        # Check custom predicate
        if self.predicate is not None:
            should_exit = self.predicate(context)
            if should_exit:
                logger.info(
                    f"[{thread_id}#{thread_ident}] DiagnosticCondition: "
                    f"predicate returned True, exiting loop"
                )
                return LoopControlAction.BREAK
        
        # Continue looping
        logger.debug(
            f"[{thread_id}#{thread_ident}] DiagnosticCondition: "
            f"continue (iteration {self.iteration_count})"
        )
        return LoopControlAction.PASS


if __name__ == "__main__":
    """Test diagnostic condition behavior."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*70)
    print("DiagnosticCondition Tests")
    print("="*70)
    
    # Test 1: Max iterations
    print("\n1. Max iterations (3):")
    cond1 = DiagnosticCondition(max_iterations=3)
    ctx = Context()
    
    for i in range(5):
        result = cond1.run(ctx)
        print(f"   Iteration {i+1}: readiness={cond1.readiness}")
        if not cond1.readiness:
            break
    assert cond1.iteration_count == 3
    print("✅ Max iterations works")
    
    # Test 2: Timeout
    print("\n2. Timeout (0.1 seconds):")
    cond2 = DiagnosticCondition(timeout_seconds=0.1)
    ctx2 = Context()
    
    iteration = 0
    while True:
        result = cond2.run(ctx2)
        iteration += 1
        print(f"   Iteration {iteration}: readiness={cond2.readiness}")
        if not cond2.readiness:
            break
        time.sleep(0.03)  # 30ms per iteration
    
    assert iteration >= 3  # Should run at least 3 iterations (90ms)
    print(f"✅ Timeout works ({iteration} iterations)")
    
    # Test 3: Custom predicate
    print("\n3. Custom predicate (stop when counter >= 5):")
    
    def stop_at_5(ctx: Context) -> bool:
        if ContextDataType.DIAGNOSTIC not in ctx.data:
            return False
        counters = {item['key']: item['value'] 
                   for item in ctx.data[ContextDataType.DIAGNOSTIC] 
                   if isinstance(item, dict) and 'key' in item}
        return counters.get('_counter_test', 0) >= 5
    
    cond3 = DiagnosticCondition(predicate=stop_at_5)
    ctx3 = Context()
    
    from src.pipeline.diagnostic_task import DiagnosticTask
    counter_task = DiagnosticTask(increment_counter='test')
    
    iteration = 0
    while True:
        ctx3 = counter_task.run(ctx3)
        result = cond3.run(ctx3)
        iteration += 1
        
        counters = {item['key']: item['value'] 
                   for item in ctx3.data[ContextDataType.DIAGNOSTIC] 
                   if isinstance(item, dict) and 'key' in item}
        count = counters.get('_counter_test', 0)
        
        print(f"   Iteration {iteration}: counter={count}, readiness={cond3.readiness}")
        if not cond3.readiness:
            break
    
    assert iteration == 5
    print("✅ Custom predicate works")
    
    # Test 4: Combined conditions (first to trigger)
    print("\n4. Combined conditions (max_iter=10, timeout=0.05s):")
    cond4 = DiagnosticCondition(max_iterations=10, timeout_seconds=0.05)
    ctx4 = Context()
    
    iteration = 0
    while True:
        result = cond4.run(ctx4)
        iteration += 1
        if not cond4.readiness:
            print(f"   Exited after {iteration} iterations")
            break
        time.sleep(0.02)  # 20ms per iteration
    
    assert iteration <= 10  # Should timeout before max iterations
    assert iteration >= 2   # Should run at least 2 iterations
    print("✅ Combined conditions work (timeout won)")
    
    print("\n" + "="*70)
    print("All DiagnosticCondition tests passed!")
    print("="*70)
