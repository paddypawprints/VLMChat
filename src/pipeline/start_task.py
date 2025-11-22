"""
Start task - empty source task for pipeline entry point.

Used as the beginning of a pipeline, especially before splits where
multiple parallel branches need a common starting point.
"""

from .task_base import BaseTask, Context, register_task


@register_task('start')
class StartTask(BaseTask):
    """
    Empty source task that produces no data.
    
    Used as an entry point for pipelines, particularly before splits
    where multiple parallel branches (e.g., camera + console input)
    need to start from the same point.
    """
    
    def __init__(self, task_id: str = "start"):
        """
        Initialize start task.
        
        Args:
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        
        # Define contracts - no inputs, no outputs (just passes context through)
        self.input_contract = {}  # Source task, no inputs
        self.output_contract = {}  # No data produced
    
    def run(self, context: Context) -> Context:
        """
        Pass through context unchanged.
        
        Args:
            context: Pipeline context (typically empty at this stage)
            
        Returns:
            Unchanged context
        """
        # No-op: just return the context as-is
        return context
