"""
Clear task for removing data from context.

Provides DSL interface to Context.clear() method for pipeline-level
data lifecycle management.
"""

import logging
from typing import Optional
from ..core.task_base import BaseTask, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('clear')
class ClearTask(BaseTask):
    """
    Task that removes data from context using Context.clear() method.
    
    Useful for explicit cleanup in pipelines, especially for managing
    memory in continuous pipelines or cleaning up intermediate data.
    
    Usage:
        # Clear everything
        clear()
        
        # Clear all images
        clear(type=image)
        
        # Clear specific label across all types
        clear(label=frame)
        
        # Clear specific type+label
        clear(type=image, label=frame)
        
        # Remove last item from label
        clear(type=image, label=frame, items=-1)
        
        # Remove first 3 items
        clear(type=image, label=frame, items=3)
    
    Examples:
        # Clean up frames after detection
        camera() -> yolo() -> clear(type=image, label=frame) -> viewer()
        
        # Keep only last frame for history
        camera() -> clear(type=image, label=frame, items=-1) -> process()
        
        # Clear all between iterations
        camera() -> process() -> viewer() -> clear() -> loop()
    """
    
    def __init__(self, 
                 data_type: Optional[ContextDataType] = None,
                 label: Optional[str] = None,
                 items: int = 0,
                 task_id: str = "clear"):
        """
        Initialize clear task.
        
        Args:
            data_type: Type to clear (None = all types)
            label: Label to clear (None = all labels)
            items: Number of items to remove (0 = all, positive = first N, negative = last N)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.data_type = data_type
        self.label = label
        self.items = items
        
        # Define contracts
        self.input_contract = {}  # Accepts any context
        self.output_contract = {}  # Returns context with data removed
    
    def configure(self, **kwargs) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            **kwargs: Configuration parameters
                - type: Data type name to clear (e.g., "image", "text")
                - label: Label to clear (e.g., "frame", "detections")
                - items: Number of items to remove (int)
                
        Example DSL:
            clear(type=image, label=frame, items=-1)
        """
        if "type" in kwargs:
            type_name = kwargs["type"]
            # Find matching ContextDataType by type_name
            for dt in ContextDataType:
                if dt.type_name == type_name:
                    self.data_type = dt
                    break
            else:
                logger.warning(f"Unknown data type: {type_name}")
        
        if "label" in kwargs:
            self.label = kwargs["label"]
        
        if "items" in kwargs:
            try:
                self.items = int(kwargs["items"])
            except ValueError:
                logger.warning(f"Invalid items value: {kwargs['items']}, using 0")
                self.items = 0
        
        logger.debug(f"ClearTask configured: type={self.data_type}, label={self.label}, items={self.items}")
    
    def run(self, context: Context) -> Context:
        """
        Clear data from context.
        
        Args:
            context: Input context to clear from
            
        Returns:
            Same context with data removed
        """
        context.clear(data_type=self.data_type, label=self.label, items=self.items)
        
        # Log what was cleared
        if self.data_type and self.label:
            if self.items == 0:
                logger.debug(f"Cleared all {self.data_type.type_name}[{self.label}]")
            elif self.items > 0:
                logger.debug(f"Cleared first {self.items} {self.data_type.type_name}[{self.label}]")
            else:
                logger.debug(f"Cleared last {abs(self.items)} {self.data_type.type_name}[{self.label}]")
        elif self.data_type:
            logger.debug(f"Cleared all {self.data_type.type_name}")
        elif self.label:
            logger.debug(f"Cleared all labels named '{self.label}'")
        else:
            logger.debug("Cleared entire context")
        
        return context
    
    def __str__(self) -> str:
        """String representation."""
        parts = []
        if self.data_type:
            parts.append(f"type={self.data_type.type_name}")
        if self.label:
            parts.append(f"label={self.label}")
        if self.items != 0:
            parts.append(f"items={self.items}")
        
        if parts:
            return f"ClearTask({', '.join(parts)})"
        return "ClearTask(clear_all)"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()
