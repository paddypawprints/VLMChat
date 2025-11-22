"""
Console output task for pipeline integration.

Displays the model response to console (sink task).
"""

from ..task_base import BaseTask, Context, ContextDataType, register_task


@register_task('console_output')
class ConsoleOutputTask(BaseTask):
    """
    Pipeline sink task that displays response to console.
    
    Takes RESPONSE from context and prints it to stdout.
    This is a side-effect task with no downstream consumers (sink task).
    """
    
    def __init__(self, prefix: str = "SmolVLM: ", task_id: str = "console_output"):
        """
        Initialize console output task.
        
        Args:
            prefix: Text to display before the response
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.prefix = prefix
        
        # Define contracts
        # Input: Requires RESPONSE
        # Output: No output (sink task)
        self.input_contract = {ContextDataType.RESPONSE: str}
        self.output_contract = {}  # Sink task produces no output
    
    def run(self, context: Context) -> Context:
        """
        Display response to console.
        
        Args:
            context: Pipeline context containing RESPONSE
            
        Returns:
            Unchanged context (sink task)
        """
        # Get response from context
        response_text = context.data.get(ContextDataType.RESPONSE)
        
        if response_text is None:
            raise ValueError(f"Task {self.task_id}: RESPONSE not found in context")
        
        # Display to console (side effect)
        print(f"\n{self.prefix}{response_text}")
        
        # Return context unchanged
        return context
