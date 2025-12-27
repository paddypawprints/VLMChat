"""
Exit code-based loop control conditions.

Provides loop conditions that check task exit codes from trace events:
- break_on: Exit loop if last task's exit code matches (default: any non-zero)
- continue_on: Retry loop if last task's exit code matches
"""

import logging
from typing import Optional, Set
from ...core.task_base import LoopCondition, LoopControlAction, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('break_on')
class BreakOnCondition(LoopCondition):
    """
    Loop control condition that breaks when previous task's exit code matches.
    
    Checks the exit code of the most recent task execution in the trace.
    Breaks loop if exit code matches the specified code(s).
    
    Exit code convention (unix-style):
    - 0: Success
    - 1: Generic failure (empty input, no data, etc.)
    - 2: Exception/error during execution
    - 3+: Task-specific error codes
    
    Parameters:
    - code: Exit code(s) to match. Can be:
      - Single integer: "code=1" (break if exit_code == 1)
      - Multiple: "code=1,2,3" (break if exit_code in [1,2,3])
      - "non-zero" or omitted: Break on any non-zero exit code
    
    Examples:
        ```python
        # Break on empty input (exit code 1)
        condition = BreakOnCondition(code=1)
        
        # Break on any failure (any non-zero)
        condition = BreakOnCondition()  # default
        
        # Break on specific codes
        condition = BreakOnCondition(code="1,3,5")
        ```
        
        DSL:
        ```
        {console_input -> break_on(code=1) -> process}
        {camera -> detect -> break_on() -> process}  # any non-zero
        {task -> break_on(code=2,3) -> next}
        ```
    """
    
    def __init__(self, code: Optional[str] = None, task_id: str = "break_on"):
        """
        Initialize break on condition.
        
        Args:
            code: Exit code(s) to match (default: any non-zero)
            task_id: Unique identifier for this condition
        """
        super().__init__(task_id)
        self.match_codes: Optional[Set[int]] = None
        
        if code is not None:
            self._parse_codes(code)
        
        logger.info(f"BreakOnCondition '{task_id}' initialized with codes={self.match_codes}")
    
    def configure(self, **params) -> None:
        """
        Configure condition from DSL parameters.
        
        Args:
            **params: Configuration parameters
                - code: Exit code(s) to match (int, comma-separated string, or "non-zero")
        """
        if "code" in params:
            self._parse_codes(params["code"])
            logger.info(f"BreakOnCondition '{self.task_id}' configured: codes={self.match_codes}")
    
    def _parse_codes(self, code_param) -> None:
        """
        Parse code parameter into set of integers.
        
        Args:
            code_param: Can be int, string number, comma-separated string, or "non-zero"
        """
        if code_param == "non-zero" or code_param is None:
            self.match_codes = None  # Match any non-zero
            return
        
        if isinstance(code_param, int):
            self.match_codes = {code_param}
            return
        
        # Parse as string
        code_str = str(code_param).strip()
        
        if code_str == "non-zero" or code_str == "":
            self.match_codes = None
            return
        
        # Parse comma-separated codes
        try:
            self.match_codes = set(int(c.strip()) for c in code_str.split(","))
        except ValueError:
            logger.error(f"Invalid code parameter: '{code_param}', using default (any non-zero)")
            self.match_codes = None
    
    def evaluate(self, context: Context) -> LoopControlAction:
        """
        Check if previous task's exit code matches criteria.
        
        Args:
            context: Pipeline context with LOOP_STACK
        
        Returns:
            BREAK if exit code matches, PASS otherwise
        """
        # Get loop stack from context
        stack = context.data.get(ContextDataType.LOOP_STACK)
        
        if not stack or len(stack) == 0:
            # Not in a loop context
            logger.warning(f"[{self.task_id}] Evaluated outside loop context")
            return LoopControlAction.PASS
        
        # Get current loop state (top of stack)
        current_loop = stack[-1]
        exit_code = current_loop.get('last_exit_code', 0)
        
        # Check if exit code matches
        should_break = False
        
        if self.match_codes is None:
            # Match any non-zero
            should_break = (exit_code != 0)
        else:
            # Match specific codes
            should_break = (exit_code in self.match_codes)
        
        if should_break:
            logger.info(f"[{self.task_id}] exit_code={exit_code} matches criteria, breaking loop")
            return LoopControlAction.BREAK
        else:
            logger.debug(f"[{self.task_id}] exit_code={exit_code} does not match, continuing")
            return LoopControlAction.PASS
    
    def describe(self) -> str:
        """Return description of what this condition does."""
        return "Breaks loop if previous task's exit code matches specified code(s). Default: any non-zero exit code."
    
    def describe_parameters(self) -> dict:
        """Return parameter descriptions."""
        return {
            "code": {
                "description": "Exit code(s) to match: single number, comma-separated list, or 'non-zero' (default)",
                "type": "str or int",
                "default": "non-zero",
                "example": "1",
                "choices": ["Any integer", "comma-separated integers", "non-zero"]
            }
        }


@register_task('continue_on')
class ContinueOnCondition(LoopCondition):
    """
    Loop control condition that restarts loop when previous task's exit code matches.
    
    Checks the exit code of the most recent task execution. If it matches the
    specified code(s), returns CONTINUE to restart the loop (retry pattern).
    
    Parameters:
    - code: Exit code(s) to match for retry. Can be:
      - Single integer: "code=1" (retry if exit_code == 1)
      - Multiple: "code=1,2" (retry if exit_code in [1,2])
      - "non-zero" or omitted: Retry on any non-zero exit code
    
    Useful for retrying operations until they succeed:
    ```dsl
    {camera -> detect -> continue_on(code=1) -> process}
    ```
    
    Examples:
        ```python
        # Retry until detection succeeds
        condition = ContinueOnCondition()
        
        # Retry only on empty result (code 1)
        condition = ContinueOnCondition(code=1)
        ```
        
        DSL:
        ```
        {camera -> detect -> continue_on() -> process}  # retry on any failure
        {input -> continue_on(code=1) -> process}  # retry only on empty input
        ```
    """
    
    def __init__(self, code: Optional[str] = None, task_id: str = "continue_on"):
        """
        Initialize continue on condition.
        
        Args:
            code: Exit code(s) to match for retry (default: any non-zero)
            task_id: Unique identifier for this condition
        """
        super().__init__(task_id)
        self.match_codes: Optional[Set[int]] = None
        
        if code is not None:
            self._parse_codes(code)
        
        logger.info(f"ContinueOnCondition '{task_id}' initialized with codes={self.match_codes}")
    
    def configure(self, **params) -> None:
        """
        Configure condition from DSL parameters.
        
        Args:
            **params: Configuration parameters
                - code: Exit code(s) to match for retry
        """
        if "code" in params:
            self._parse_codes(params["code"])
            logger.info(f"ContinueOnCondition '{self.task_id}' configured: codes={self.match_codes}")
    
    def _parse_codes(self, code_param) -> None:
        """Parse code parameter into set of integers."""
        if code_param == "non-zero" or code_param is None:
            self.match_codes = None
            return
        
        if isinstance(code_param, int):
            self.match_codes = {code_param}
            return
        
        code_str = str(code_param).strip()
        
        if code_str == "non-zero" or code_str == "":
            self.match_codes = None
            return
        
        try:
            self.match_codes = set(int(c.strip()) for c in code_str.split(","))
        except ValueError:
            logger.error(f"Invalid code parameter: '{code_param}', using default (any non-zero)")
            self.match_codes = None
    
    def evaluate(self, context: Context) -> LoopControlAction:
        """
        Check if previous task's exit code matches retry criteria.
        
        Args:
            context: Pipeline context with LOOP_STACK
        
        Returns:
            CONTINUE if exit code matches, PASS otherwise
        """
        # Get loop stack from context
        stack = context.data.get(ContextDataType.LOOP_STACK)
        
        if not stack or len(stack) == 0:
            return LoopControlAction.PASS
        
        # Get current loop state
        current_loop = stack[-1]
        exit_code = current_loop.get('last_exit_code', 0)
        
        # Check if exit code matches
        should_retry = False
        
        if self.match_codes is None:
            # Match any non-zero
            should_retry = (exit_code != 0)
        else:
            # Match specific codes
            should_retry = (exit_code in self.match_codes)
        
        if should_retry:
            logger.info(f"[{self.task_id}] exit_code={exit_code} matches, retrying")
            return LoopControlAction.CONTINUE
        else:
            logger.debug(f"[{self.task_id}] exit_code={exit_code} does not match, continuing")
            return LoopControlAction.PASS
    
    def describe(self) -> str:
        """Return description of what this condition does."""
        return "Restarts loop if previous task's exit code matches specified code(s). Default: any non-zero exit code."
    
    def describe_parameters(self) -> dict:
        """Return parameter descriptions."""
        return {
            "code": {
                "description": "Exit code(s) to match for retry: single number, comma-separated list, or 'non-zero' (default)",
                "type": "str or int",
                "default": "non-zero",
                "example": "1",
                "choices": ["Any integer", "comma-separated integers", "non-zero"]
            }
        }

    """
    Loop control condition that breaks when previous task fails.
    
    Checks the exit code of the most recent task execution in the trace.
    If exit_code != 0, returns BREAK to exit the loop.
    
    This allows simple error handling without custom conditions:
    ```dsl
    {console_input -> break_on_fail -> process}
    ```
    
    Exit code convention (unix-style):
    - 0: Success
    - 1: Generic failure (empty input, no data, etc.)
    - 2: Exception/error during execution
    - 3+: Task-specific error codes
    
    Example:
        ```python
        # Exit loop if console_input returns empty string
        condition = BreakOnFailCondition()
        ```
        
        DSL:
        ```
        {console_input(prompt="Enter command:") -> break_on_fail -> process}
        ```
    """
    
    def __init__(self, task_id: str = "break_on_fail"):
        """
        Initialize break on fail condition.
        
        Args:
            task_id: Unique identifier for this condition
        """
        super().__init__(task_id)
        logger.info(f"BreakOnFailCondition '{task_id}' initialized")
    
    def evaluate(self, context: Context) -> LoopControlAction:
        """
        Check if previous task failed (exit_code != 0).
        
        Looks at the most recent 'execute' event in context trace and
        checks its exit code. If non-zero, returns BREAK.
        
        Args:
            context: Pipeline context with trace events
        
        Returns:
            BREAK if last task failed, PASS otherwise
        """
        # Get trace events from context
        if ContextDataType.TRACE not in context.data:
            logger.warning(f"[{self.task_id}] No trace data in context")
            return LoopControlAction.PASS
        
        trace_events = context.data[ContextDataType.TRACE]
        if not trace_events:
            logger.warning(f"[{self.task_id}] Empty trace events list")
            return LoopControlAction.PASS
        
        # Find most recent 'execute' event
        # Trace format: (timestamp, thread_id, task_id, upstream_ids, event_type, submission_time, exit_code)
        execute_events = [e for e in trace_events if len(e) >= 7 and e[4] == 'execute']
        
        if not execute_events:
            logger.warning(f"[{self.task_id}] No execute events in trace")
            return LoopControlAction.PASS
        
        # Get most recent execute event (sorted by timestamp)
        last_event = sorted(execute_events, key=lambda e: e[0])[-1]
        last_task_id = last_event[2]
        exit_code = last_event[6]
        
        # Skip if the last event is this condition itself
        if last_task_id == self.task_id:
            # Look at second-to-last event
            if len(execute_events) >= 2:
                last_event = sorted(execute_events, key=lambda e: e[0])[-2]
                last_task_id = last_event[2]
                exit_code = last_event[6]
            else:
                logger.debug(f"[{self.task_id}] Only condition event in trace, passing")
                return LoopControlAction.PASS
        
        if exit_code != 0:
            # Task failed, break loop
            logger.info(f"[{self.task_id}] Task '{last_task_id}' failed with exit_code={exit_code}, breaking loop")
            return LoopControlAction.BREAK
        else:
            # Task succeeded, continue
            logger.debug(f"[{self.task_id}] Task '{last_task_id}' succeeded (exit_code=0), continuing")
            return LoopControlAction.PASS
    
    def describe(self) -> str:
        """Return description of what this condition does."""
        return "Breaks loop if previous task failed (exit_code != 0). Checks most recent task execution in trace."
    
    def describe_parameters(self) -> dict:
        """Return parameter descriptions (no parameters for this condition)."""
        return {}


@register_task('continue_on_fail')
class ContinueOnFailCondition(LoopCondition):
    """
    Loop control condition that restarts loop when previous task fails.
    
    Checks the exit code of the most recent task execution. If exit_code != 0,
    returns CONTINUE to restart the loop (retry pattern).
    
    Useful for retrying operations until they succeed:
    ```dsl
    {camera -> detect -> continue_on_fail -> process}
    ```
    
    Example:
        ```python
        # Retry until detection succeeds
        condition = ContinueOnFailCondition()
        ```
        
        DSL:
        ```
        {camera -> detect -> continue_on_fail -> process}
        ```
    """
    
    def __init__(self, task_id: str = "continue_on_fail"):
        """
        Initialize continue on fail condition.
        
        Args:
            task_id: Unique identifier for this condition
        """
        super().__init__(task_id)
        logger.info(f"ContinueOnFailCondition '{task_id}' initialized")
    
    def evaluate(self, context: Context) -> LoopControlAction:
        """
        Check if previous task failed and should retry.
        
        Args:
            context: Pipeline context with trace events
        
        Returns:
            CONTINUE if last task failed, PASS otherwise
        """
        # Get trace events from context
        if ContextDataType.TRACE not in context.data:
            return LoopControlAction.PASS
        
        trace_events = context.data[ContextDataType.TRACE]
        if not trace_events:
            return LoopControlAction.PASS
        
        # Find most recent 'execute' event
        execute_events = [e for e in trace_events if len(e) >= 7 and e[4] == 'execute']
        
        if not execute_events:
            return LoopControlAction.PASS
        
        # Get most recent execute event
        last_event = sorted(execute_events, key=lambda e: e[0])[-1]
        last_task_id = last_event[2]
        exit_code = last_event[6]
        
        # Skip if the last event is this condition itself
        if last_task_id == self.task_id:
            if len(execute_events) >= 2:
                last_event = sorted(execute_events, key=lambda e: e[0])[-2]
                last_task_id = last_event[2]
                exit_code = last_event[6]
            else:
                return LoopControlAction.PASS
        
        if exit_code != 0:
            # Task failed, retry
            logger.info(f"[{self.task_id}] Task '{last_task_id}' failed with exit_code={exit_code}, retrying")
            return LoopControlAction.CONTINUE
        else:
            # Task succeeded, continue to next task
            logger.debug(f"[{self.task_id}] Task '{last_task_id}' succeeded, continuing")
            return LoopControlAction.PASS
    
    def describe(self) -> str:
        """Return description of what this condition does."""
        return "Restarts loop if previous task's exit code matches specified code(s). Default: any non-zero exit code."
    
    def describe_parameters(self) -> dict:
        """Return parameter descriptions."""
        return {
            "code": {
                "description": "Exit code(s) to match for retry: single number, comma-separated list, or 'non-zero' (default)",
                "type": "str or int",
                "default": "non-zero",
                "example": "1",
                "choices": ["Any integer", "comma-separated integers", "non-zero"]
            }
        }
