"""
Console input task adapter for pipeline integration.

Captures user text input from console and stores it in the pipeline context.
"""

from typing import Dict
from ..task_base import BaseTask, Context, ContextDataType, register_task


@register_task('console_input')
class ConsoleInputTask(BaseTask):
    """
    Pipeline task that captures text input from console.
    
    Blocks until user enters text, then stores it in context as PROMPT.
    This task can run in parallel with other tasks (e.g., camera capture)
    in a split pipeline.
    """
    
    def __init__(self, prompt_text: str = "You: ", task_id: str = "console_input"):
        """
        Initialize console input task.
        
        Args:
            prompt_text: Text to display when requesting input
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.prompt_text = prompt_text
        
        # Define contracts - no inputs, produces PROMPT
        self.input_contract = {}  # Can be source or follow start task
        self.output_contract = {ContextDataType.PROMPT: str}
    
    def configure(self, params: Dict[str, str]) -> None:
        """
        Configure console input from parameters (DSL support).
        
        Args:
            params: Dictionary with configuration
                - prompt: Text to display (e.g., "Enter command: ")
        
        Example:
            task.configure({"prompt": "Enter command: "})
        """
        if "prompt" in params:
            self.prompt_text = params["prompt"]
    
    def run(self, context: Context) -> Context:
        """
        Capture text input from console and store in context.
        
        This method blocks until user enters text and presses Enter.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with PROMPT added
        """
        # Capture user input (blocking)
        try:
            user_input = input(self.prompt_text).strip()
        except EOFError:
            user_input = ""
        
        # Store in context
        context.data[ContextDataType.PROMPT] = user_input
        
        return context
