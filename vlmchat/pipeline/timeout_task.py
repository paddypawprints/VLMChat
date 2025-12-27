"""
Timeout condition for loop exit control.

Breaks loop execution after a specified time duration.
"""

import time
import logging
from typing import Optional
from .task_base import BaseTask, Context, ContextDataType, LoopCondition, LoopControlAction, register_task

logger = logging.getLogger(__name__)


@register_task('timeout')
class TimeoutCondition(LoopCondition):
    """
    Loop control condition that breaks after N seconds.
    
    Checks elapsed time against timeout duration and returns BREAK
    when timeout is reached.
    
    Attributes:
        timeout_seconds: Duration in seconds before breaking loop
    
    Example:
        ```python
        # Exit loop after 60 seconds
        timeout = TimeoutCondition(timeout_seconds=60.0)
        ```
        
        DSL:
        ```
        {camera -> process ->:timeout(60)}
        ```
    """
    
    def __init__(self, timeout_seconds: float = 60.0, task_id: str = "timeout"):
        """
        Initialize timeout condition.
        
        Args:
            timeout_seconds: Duration in seconds before breaking loop (default: 60.0)
            task_id: Unique identifier for this condition
        """
        super().__init__(task_id)
        self.timeout = timeout_seconds
        logger.info(f"TimeoutCondition '{task_id}' initialized with timeout={timeout_seconds}s")
    
    def evaluate(self, context: Context) -> LoopControlAction:
        """
        Check if timeout has been reached.
        
        Args:
            context: Pipeline context with LOOP_STACK
        
        Returns:
            BREAK if timeout reached, PASS otherwise
        """
        # Get loop stack from context
        stack = context.data.get(ContextDataType.LOOP_STACK)
        
        if not stack or len(stack) == 0:
            # Not in a loop context
            logger.warning(f"TimeoutCondition '{self.task_id}' evaluated outside loop")
            return LoopControlAction.PASS
        
        # Get current loop state (top of stack)
        current_loop = stack[-1]
        start_time = current_loop.get('start_time')
        
        if start_time is None:
            logger.error(f"Loop state missing start_time")
            return LoopControlAction.PASS
        
        # Calculate elapsed time
        elapsed = time.time() - start_time
        
        if elapsed >= self.timeout:
            # Time's up, signal loop exit
            iteration = current_loop.get('iteration', 0)
            logger.info(f"[{self.task_id}] Timeout reached: {elapsed:.1f}s >= {self.timeout}s "
                       f"(after {iteration} iterations)")
            return LoopControlAction.BREAK
        else:
            # Continue looping
            logger.debug(f"[{self.task_id}] {elapsed:.1f}s / {self.timeout}s elapsed")
            return LoopControlAction.PASS
    
    def configure(self, **kwargs) -> None:
        """
        Configure timeout from parameters.
        
        Args:
            **kwargs: Configuration parameters with 'timeout' or 'seconds' key (float seconds)
        """
        if 'timeout' in kwargs:
            self.timeout = float(kwargs['timeout'])
            logger.info(f"TimeoutCondition '{self.task_id}' configured: timeout={self.timeout}s")
        elif 'seconds' in kwargs:
            self.timeout = float(kwargs['seconds'])
            logger.info(f"TimeoutCondition '{self.task_id}' configured: timeout={self.timeout}s")


# Backward compatibility alias
TimeoutTask = TimeoutCondition

