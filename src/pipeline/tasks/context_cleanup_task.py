"""
Context cleanup task for managing pipeline context lifecycle.

Removes specified data types from context to prevent accumulation in loops.
"""

import logging
from typing import Dict, Any, List
from ..task_base import BaseTask, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('cleanup')
class ContextCleanupTask(BaseTask):
    """
    Pipeline task that removes specified data types from context.
    
    Useful for cleaning up context between loop iterations to prevent
    data accumulation. Can remove specific types like TEXT, DETECTIONS, etc.
    """
    
    def __init__(self, task_id: str = "context_cleanup"):
        """
        Initialize context cleanup task.
        
        Args:
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.remove_types: List[str] = []
        
        # Define contracts - no specific inputs/outputs (context manipulation)
        self.input_contract = {}
        self.output_contract = {}
    
    def configure(self, **params) -> None:
        """
        Configure which context types to remove.
        
        Args:
            **params: Keyword arguments with configuration
                - remove_types: Comma-separated string or list of types to remove
                                Supported: "text", "image", "detections", "prompt_embeddings"
        
        Example:
            task.configure(remove_types="text")
            task.configure(remove_types="text,detections")
        """
        if "remove_types" in params:
            types_param = params["remove_types"]
            
            # Handle comma-separated string or list
            if isinstance(types_param, str):
                self.remove_types = [t.strip().lower() for t in types_param.split(",")]
            elif isinstance(types_param, list):
                self.remove_types = [str(t).strip().lower() for t in types_param]
            else:
                raise ValueError(f"Task {self.task_id}: remove_types must be string or list")
    
    def run(self, context: Context) -> Context:
        """
        Remove specified data types from context.
        
        Args:
            context: Pipeline context
            
        Returns:
            Updated context with specified types removed
        """
        # Map string names to ContextDataType enum values
        type_map = {
            "text": ContextDataType.TEXT,
            "image": ContextDataType.IMAGE,
            "detections": ContextDataType.DETECTIONS,
            "prompt_embeddings": ContextDataType.PROMPT_EMBEDDINGS,
        }
        
        # Remove each specified type
        for type_name in self.remove_types:
            if type_name not in type_map:
                logger.warning(f"Task {self.task_id}: Unknown type '{type_name}', skipping")
                continue
            
            context_type = type_map[type_name]
            
            # Remove from context data
            if context_type in context.data:
                removed_count = len(context.data[context_type]) if isinstance(context.data[context_type], list) else 1
                del context.data[context_type]
                logger.debug(f"Task {self.task_id}: Removed {removed_count} {type_name} item(s)")
        
        return context
    
    def describe(self) -> str:
        """Return description of what this task does."""
        return ("Removes specified data types from pipeline context. "
                "Useful for preventing data accumulation in loops.")
    
    def describe_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter descriptions for context cleanup configuration."""
        return {
            "remove_types": {
                "description": "Comma-separated types to remove: text, image, detections, prompt_embeddings",
                "type": "str or list",
                "required": True,
                "example": "text,detections"
            }
        }
