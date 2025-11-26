"""
Console input task adapter for pipeline integration.

Captures user text input from console and stores it in the pipeline context.
"""

from typing import Dict, Any
from ..task_base import BaseTask, Context, ContextDataType, register_task


@register_task('input')
class ConsoleInputTask(BaseTask):
    """
    Pipeline task that captures text input from console or uses provided text.
    
    Can operate in two modes:
    1. Interactive: Blocks and waits for user input from console
    2. Non-interactive: Uses provided text value (no blocking)
    
    TEXT flows through the pipeline like unix stdout, allowing tasks
    to process and transform text through multiple stages.
    """
    
    def __init__(self, task_id: str = "console_input"):
        """
        Initialize console input task.
        
        Args:
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.input_value = None  # If set, use this instead of reading console
        
        # Define contracts - no inputs, produces TEXT
        self.input_contract = {}  # Can be source or follow start task
        self.output_contract = {ContextDataType.TEXT: str}
    
    def configure(self, **params) -> None:
        """
        Configure console input from parameters (DSL support).
        
        Args:
            **params: Keyword arguments with configuration
                - input: Pre-set text value (non-interactive mode, no console read)
        
        Example:
            task.configure(input="test message")  # Non-interactive
        """
        if "input" in params:
            self.input_value = params["input"]
    
    def run(self, context: Context) -> Context:
        """
        Capture text input from console or use provided value.
        
        If 'input' parameter was configured, uses that value (non-interactive).
        Otherwise, gets text from pipeline runner's input queue (environment-agnostic).
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with TEXT appended
        """
        # Get text: either from pre-configured value or pipeline runner's queue
        if self.input_value is not None:
            # Non-interactive mode: use provided value
            user_input = self.input_value
        else:
            # Queue mode: get input from pipeline runner (blocks with timeout)
            runner = getattr(context, 'pipeline_runner', None)
            if runner is None:
                raise RuntimeError(f"Task {self.task_id}: context.pipeline_runner not set - cannot get input")
            
            try:
                # Block for up to 60 seconds waiting for input
                user_input = runner.input_queue.get(timeout=60)
            except Exception as e:
                # Timeout or error - treat as empty input
                user_input = ""
        
        # Store in context as TEXT (flows like unix stdout)
        if ContextDataType.TEXT not in context.data:
            context.data[ContextDataType.TEXT] = []
        context.data[ContextDataType.TEXT].append(user_input)
        
        # Set exit code: 0 for success (non-empty), 1 for empty input
        self.exit_code = 0 if user_input else 1
        
        return context
    
    def describe(self) -> str:
        """Return description of what this task does."""
        return ("Captures text input from console (interactive) or uses provided text value (non-interactive). "
                "Appends text to pipeline context. Exit codes: 0=non-empty input (success), 1=empty input (failure).")
    
    def describe_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter descriptions for console input configuration."""
        return {
            "input": {
                "description": "Pre-set text value (non-interactive mode, skips console read)",
                "type": "str",
                "example": "test message"
            }
        }
    
    def describe_exit_codes(self) -> Dict[int, str]:
        """Return exit code descriptions."""
        return {
            0: "Success: non-empty input received",
            1: "Failure: empty input (user pressed Enter without text)"
        }
