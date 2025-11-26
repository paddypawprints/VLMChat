"""
Test input task for non-interactive testing and automation.

Provides pre-configured text prompts without blocking for user input.
"""

from typing import Dict, Any, List
from ..task_base import BaseTask, Context, ContextDataType, register_task


@register_task('test_input')
class TestInputTask(BaseTask):
    """
    Non-interactive input task that provides pre-configured prompts.
    
    Designed for testing and automation where interactive console input
    is not available or desired. Prompts are configured via the 'prompts'
    parameter and returned sequentially on each run.
    
    When all prompts are exhausted, returns empty string and sets exit_code=1
    to signal completion (works with break_on connector).
    
    Example DSL:
        test_input(prompts="person,horse,cowboy")
        
    Example in loop:
        {test_input(prompts="a,b,c") -> :break_on(code=1):}
    """
    
    def __init__(self, task_id: str = "test_input"):
        """
        Initialize test input task.
        
        Args:
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.prompts: List[str] = []
        self.current_index = 0
        
        # Define contracts - no inputs, produces TEXT
        self.input_contract = {}
        self.output_contract = {ContextDataType.TEXT: str}
    
    def configure(self, **params) -> None:
        """
        Configure test input from parameters (DSL support).
        
        Args:
            **params: Keyword arguments with configuration
                - prompts: Comma-separated string or list of prompts
        
        Example:
            task.configure(prompts="person,horse,cowboy")
            task.configure(prompts=["person", "horse", "cowboy"])
        """
        if "prompts" in params:
            prompts_param = params["prompts"]
            if isinstance(prompts_param, str):
                # Split comma-separated string
                self.prompts = [p.strip() for p in prompts_param.split(',') if p.strip()]
            elif isinstance(prompts_param, list):
                self.prompts = prompts_param
            else:
                self.prompts = [str(prompts_param)]
    
    def run(self, context: Context) -> Context:
        """
        Return all prompts at once in TEXT list.
        
        Previously returned prompts sequentially, but for CLIP comparison
        we need all prompts available simultaneously.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with all prompts in TEXT
        """
        # Provide all prompts at once
        if ContextDataType.TEXT not in context.data:
            context.data[ContextDataType.TEXT] = []
        
        # Add all prompts to TEXT
        context.data[ContextDataType.TEXT].extend(self.prompts)
        
        # Set exit code (0 = success)
        context.exit_code = 0
        
        return context
    
    def describe(self) -> str:
        """Return description of what this task does."""
        return "Non-interactive input task that provides pre-configured prompts for testing."
    
    def describe_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter descriptions."""
        return {
            "prompts": {
                "description": "Comma-separated string or list of text prompts to provide",
                "type": "str or list",
                "default": [],
                "example": "person,horse,cowboy"
            }
        }
