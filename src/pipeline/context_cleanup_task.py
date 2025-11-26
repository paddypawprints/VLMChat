"""
Context cleanup task for managing context lifecycle in continuous pipelines.

This task filters context data at the end of a pipeline iteration, removing
mutable data while preserving immutable data for the next iteration.
"""

import logging
from typing import List, Optional
from .task_base import BaseTask, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('context_cleanup')
class ContextCleanupTask(BaseTask):
    """
    End-of-pipeline task that filters context for next iteration.
    
    Removes mutable data (IMAGE, DETECTIONS, etc.) while keeping immutable
    data (PROMPT_EMBEDDINGS, etc.) to avoid recomputation across iterations.
    
    This is useful for continuous pipelines where some data should persist
    between runs (e.g., prompt embeddings that change infrequently) while
    other data should be discarded (e.g., camera frames, detections).
    
    Usage:
        # Automatic mode - keeps all immutable types
        cleanup = ContextCleanupTask()
        
        # Explicit mode - specify what to keep
        cleanup = ContextCleanupTask(keep_types=[
            ContextDataType.PROMPT_EMBEDDINGS
        ])
        
        # In continuous pipeline:
        context = Context()
        while True:
            context = pipeline.run(context)  # cleanup is last task
            # context now has only immutable data
    """
    
    def __init__(self, 
                 keep_types: Optional[List[ContextDataType]] = None,
                 auto_keep_immutable: bool = True,
                 task_id: str = "context_cleanup"):
        """
        Initialize context cleanup task.
        
        Args:
            keep_types: Explicit list of types to keep. If None and
                       auto_keep_immutable=True, keeps all immutable types.
            auto_keep_immutable: If True and keep_types is None, automatically
                                keeps all immutable data types.
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.keep_types = keep_types
        self.auto_keep_immutable = auto_keep_immutable
        
        # Define contracts
        self.input_contract = {}  # Accepts any context
        self.output_contract = {}  # Returns filtered context
    
    def configure(self, **kwargs) -> None:
        """
        Configure from DSL parameters.
        
        Args:
            **kwargs: Configuration parameters
                - keep_types: Comma-separated list of type names to keep
                
        Example:
            {"keep_types": "prompt_embeddings,audit"}
        """
        if "keep_types" in kwargs:
            type_names = [t.strip() for t in kwargs["keep_types"].split(",")]
            self.keep_types = []
            for name in type_names:
                # Find matching ContextDataType by type_name
                for dt in ContextDataType:
                    if dt.type_name == name:
                        self.keep_types.append(dt)
                        break
            logger.info(f"ContextCleanupTask configured to keep: {self.keep_types}")
    
    def run(self, context: Context) -> Context:
        """
        Filters context, keeping only specified data types.
        
        Args:
            context: Input context to filter
            
        Returns:
            Cleaned context with only kept data types
        """
        cleaned_context = Context()
        cleaned_context.collector = context.collector
        
        # Determine what to keep
        if self.keep_types is not None:
            # Explicit list
            types_to_keep = self.keep_types
        elif self.auto_keep_immutable:
            # All types (mutable/immutable distinction removed)
            types_to_keep = list(ContextDataType)
        else:
            # Keep nothing
            types_to_keep = []
        
        # Copy kept types (immutable, so reference is safe)
        kept = []
        removed = []
        
        for data_type, value in context.data.items():
            if data_type in types_to_keep:
                cleaned_context.data[data_type] = value
                kept.append(data_type.type_name)
            else:
                removed.append(data_type.type_name)
        
        # Log cleanup
        if kept:
            logger.debug(f"Context cleanup: kept {kept}")
        if removed:
            logger.debug(f"Context cleanup: removed {removed}")
        
        return cleaned_context
    
    def __str__(self) -> str:
        """String representation."""
        if self.keep_types:
            keep_names = [dt.type_name for dt in self.keep_types]
            return f"ContextCleanupTask(keep={keep_names})"
        elif self.auto_keep_immutable:
            return "ContextCleanupTask(auto_keep_immutable=True)"
        else:
            return "ContextCleanupTask(keep_nothing)"
    
    def __repr__(self) -> str:
        """Debug representation."""
        return self.__str__()


if __name__ == "__main__":
    # Example usage
    print("\n--- ContextCleanupTask Example ---\n")
    
    # Create a context with various data types
    ctx = Context()
    ctx.data[ContextDataType.IMAGE] = ["test_image.jpg"]
    ctx.data[ContextDataType.DETECTIONS] = [{"category": "person", "conf": 0.9}]
    ctx.data[ContextDataType.EMBEDDINGS] = [[0.1, 0.2, 0.3]]
    ctx.data[ContextDataType.PROMPT_EMBEDDINGS] = {
        "prompts": ["test prompt"],
        "embeddings": [[0.4, 0.5, 0.6]],
        "version": 1
    }
    ctx.data[ContextDataType.PROMPT] = ["test prompt text"]
    
    print(f"Original context has {len(ctx.data)} data types:")
    for dt in ctx.data.keys():
        print(f"  - {dt.type_name}")
    
    # Test 1: Auto keep immutable
    print("\n--- Test 1: Auto keep immutable ---")
    cleanup1 = ContextCleanupTask(auto_keep_immutable=True)
    print(f"Task: {cleanup1}")
    result1 = cleanup1.run(ctx)
    print(f"Result has {len(result1.data)} data types:")
    for dt in result1.data.keys():
        print(f"  - {dt.type_name}")
    
    # Test 2: Explicit keep list
    print("\n--- Test 2: Explicit keep list ---")
    cleanup2 = ContextCleanupTask(keep_types=[ContextDataType.PROMPT_EMBEDDINGS])
    print(f"Task: {cleanup2}")
    result2 = cleanup2.run(ctx)
    print(f"Result has {len(result2.data)} data types:")
    for dt in result2.data.keys():
        print(f"  - {dt.type_name}")
    
    # Test 3: Keep nothing
    print("\n--- Test 3: Keep nothing ---")
    cleanup3 = ContextCleanupTask(keep_types=[], auto_keep_immutable=False)
    print(f"Task: {cleanup3}")
    result3 = cleanup3.run(ctx)
    print(f"Result has {len(result3.data)} data types")
    
    # Test 4: Configure from params
    print("\n--- Test 4: Configure from DSL params ---")
    cleanup4 = ContextCleanupTask()
    cleanup4.configure({"keep_types": "prompt_embeddings,prompt"})
    print(f"Task: {cleanup4}")
    result4 = cleanup4.run(ctx)
    print(f"Result has {len(result4.data)} data types:")
    for dt in result4.data.keys():
        print(f"  - {dt.type_name}")
    
    print("\n✓ ContextCleanupTask tests complete")
