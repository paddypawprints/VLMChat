"""
History update task for pipeline integration.

Updates conversation history with the latest prompt and response (sink task).
"""

from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...prompt.history import History


@register_task('history_update')
class HistoryUpdateTask(BaseTask):
    """
    Pipeline sink task that updates conversation history.
    
    Takes PROMPT and RESPONSE from context and adds them to the
    conversation history. This is a side-effect task with no downstream
    consumers (sink task).
    """
    
    def __init__(self, history: History, task_id: str = "history_update"):
        """
        Initialize history update task with injected history manager.
        
        Args:
            history: History manager instance to update
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.history = history
        
        # Define contracts
        # Input: Requires PROMPT and RESPONSE
        # Output: No output (sink task)
        self.input_contract = {
            ContextDataType.PROMPT: str,
            ContextDataType.RESPONSE: str
        }
        self.output_contract = {}  # Sink task produces no output
    
    def run(self, context: Context) -> Context:
        """
        Update conversation history with prompt and response.
        
        Args:
            context: Pipeline context containing PROMPT and RESPONSE
            
        Returns:
            Unchanged context (sink task)
        """
        # Get inputs from context
        prompt_text = context.data.get(ContextDataType.PROMPT)
        response_text = context.data.get(ContextDataType.RESPONSE)
        
        if prompt_text is None:
            raise ValueError(f"Task {self.task_id}: PROMPT not found in context")
        if response_text is None:
            raise ValueError(f"Task {self.task_id}: RESPONSE not found in context")
        
        # Update history (side effect)
        self.history.add_conversation_pair(
            request_text=prompt_text,
            response_text=response_text
        )
        
        # Return context unchanged
        return context
