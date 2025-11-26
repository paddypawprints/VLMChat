"""
Detection filter task for filtering detections by category.

This task filters the DETECTIONS list to keep only specific categories.
Useful for focusing on relevant objects before clustering or encoding.
"""

from typing import List, Set
from PIL import Image

from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...object_detector.detection_base import Detection


@register_task('filter')
@register_task('detection_filter')
class DetectionFilterTask(BaseTask):
    """
    Filters detections to keep only specified categories.
    
    DSL Usage:
        filter(categories="person,car,bicycle")
        filter(categories="person,horse")
    
    Input: DETECTIONS list
    Output: Filtered DETECTIONS list
    """
    
    def __init__(self, categories: List[str] = None, task_id: str = "filter"):
        """
        Initialize detection filter.
        
        Args:
            categories: List of category names to keep (e.g., ["person", "car"])
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.categories: Set[str] = set(categories) if categories else set()
        
        # Define contracts
        self.input_contract = {ContextDataType.DETECTIONS: list}
        self.output_contract = {ContextDataType.DETECTIONS: list}
    
    def configure(self, **kwargs) -> None:
        """
        Configure filter from DSL parameters.
        
        Args:
            categories: Comma-separated category names (e.g., "person,car,bicycle")
        """
        if "categories" in kwargs:
            categories_str = kwargs["categories"]
            self.categories = set(cat.strip() for cat in categories_str.split(","))
    
    def run(self, context: Context) -> Context:
        """
        Filter detections by category.
        
        Args:
            context: Pipeline context containing DETECTIONS
            
        Returns:
            Updated context with filtered DETECTIONS
        """
        detections = context.data.get(ContextDataType.DETECTIONS, [])
        
        if not detections:
            return context
        
        if not self.categories:
            # No filter configured, pass through all detections
            return context
        
        # Filter to keep only specified categories
        filtered_detections = [
            det for det in detections 
            if det.object_category in self.categories
        ]
        
        print(f"DetectionFilter: {len(detections)} -> {len(filtered_detections)} "
              f"(kept: {', '.join(sorted(self.categories))})")
        
        context.data[ContextDataType.DETECTIONS] = filtered_detections
        return context
    
    def describe(self) -> str:
        """Return description of this task."""
        if self.categories:
            cats = ', '.join(sorted(self.categories))
            return f"Filter detections to keep only: {cats}"
        return "Filter detections by category (no categories configured)"
    
    def describe_parameters(self) -> str:
        """Return description of available parameters."""
        return """Parameters:
  categories - Comma-separated list of category names to keep
  
Examples:
  filter(categories="person,car,bicycle")
  filter(categories="person,horse,dog")"""
