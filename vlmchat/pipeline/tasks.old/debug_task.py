"""
Debug task for pipeline development and troubleshooting.

Prints context data and trace information to console for inspection.
"""

from typing import Dict, Any
from ..task_base import BaseTask, Context, ContextDataType, register_task


@register_task('debug')
class DebugTask(BaseTask):
    """
    Pipeline task that prints context and trace information for debugging.
    
    This is a pass-through task that displays context contents and optionally
    trace information to help with pipeline development and troubleshooting.
    All data flows through unchanged.
    """
    
    def __init__(self, task_id: str = "debug"):
        """
        Initialize debug task.
        
        Args:
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.show_context = True
        self.show_trace = False
        self.show_data_types = None  # None = all, or list of type names
        self.label = ""  # Optional label for this debug point
        
        # Define contracts - accepts everything, produces nothing new
        self.input_contract = {}  # Accepts any inputs
        self.output_contract = {}  # Pass-through, no new outputs
    
    def configure(self, **params) -> None:
        """
        Configure debug output from parameters (DSL support).
        
        Args:
            **params: Keyword arguments with configuration
                - context: Show context data (default: True)
                - trace: Show trace information (default: False)
                - types: Comma-separated list of data types to show (default: "all")
                - label: Optional label to identify this debug point
        
        Example:
            task.configure(label="After camera", types="image,text")
            task.configure(context=False, trace=True)
        """
        if "context" in params:
            val = params["context"]
            self.show_context = val if isinstance(val, bool) else str(val).lower() in ("true", "1", "yes")
        
        if "trace" in params:
            val = params["trace"]
            self.show_trace = val if isinstance(val, bool) else str(val).lower() in ("true", "1", "yes")
        
        if "types" in params:
            types_str = params["types"].lower()
            if types_str == "all":
                self.show_data_types = None
            else:
                self.show_data_types = [t.strip() for t in types_str.split(",")]
        
        if "label" in params:
            self.label = params["label"]
    
    def run(self, context: Context) -> Context:
        """
        Print debug information and return context unchanged.
        
        Args:
            context: Pipeline context
            
        Returns:
            Unchanged context (pass-through)
        """
        # Print separator
        print("\n" + "=" * 70)
        if self.label:
            print(f"DEBUG: {self.label} (task: {self.task_id})")
        else:
            print(f"DEBUG: {self.task_id}")
        print("=" * 70)
        
        # Show context data
        if self.show_context:
            print("\nCONTEXT DATA:")
            if not context.data:
                print("  (empty)")
            else:
                for data_type, data_list in context.data.items():
                    # Filter by types if specified
                    if self.show_data_types is not None:
                        if data_type.type_name not in self.show_data_types:
                            continue
                    
                    print(f"\n  {data_type.type_name}:")
                    if not data_list:
                        print("    (no entries)")
                    else:
                        for idx, item in enumerate(data_list):
                            # Format the item for display
                            item_str = self._format_item(item, data_type)
                            print(f"    [{idx}] {item_str}")
        
        # Show trace information
        if self.show_trace:
            print("\nTRACE INFORMATION:")
            if hasattr(context, 'trace') and context.trace:
                for trace_entry in context.trace:
                    print(f"  {trace_entry}")
            else:
                print("  (no trace data)")
        
        # Show metadata if present
        if hasattr(context, 'metadata') and context.metadata:
            print("\nMETADATA:")
            for key, value in context.metadata.items():
                print(f"  {key}: {value}")
        
        print("=" * 70 + "\n")
        
        # Return context unchanged
        return context
    
    def _format_item(self, item: Any, data_type: ContextDataType) -> str:
        """
        Format a context item for display.
        
        Args:
            item: The item to format
            data_type: The context data type
            
        Returns:
            Formatted string representation
        """
        # Handle different types
        if data_type.type_name == "text":
            # Show text with length, truncate if long
            if len(item) > 100:
                return f'"{item[:100]}..." (len={len(item)})'
            return f'"{item}"'
        
        elif data_type.type_name == "audit":
            # Show full audit log without truncation
            return f"\n{item}"
        
        elif data_type.type_name == "image":
            # Show image dimensions
            if hasattr(item, 'size'):
                return f"Image{item.size} mode={item.mode}"
            elif hasattr(item, 'shape'):
                return f"Image{item.shape}"
            return f"Image ({type(item).__name__})"
        
        elif data_type.type_name == "detections":
            # Show detection count
            if isinstance(item, list):
                return f"{len(item)} detections"
            return f"Detections ({type(item).__name__})"
        
        elif data_type.type_name == "embeddings":
            # Show embedding dimensions
            if hasattr(item, 'shape'):
                return f"Embeddings{item.shape}"
            elif isinstance(item, list):
                return f"Embeddings[{len(item)}]"
            return f"Embeddings ({type(item).__name__})"
        
        else:
            # Generic representation
            try:
                item_repr = repr(item)
                if len(item_repr) > 100:
                    return f"{item_repr[:100]}... ({type(item).__name__})"
                return item_repr
            except:
                return f"<{type(item).__name__}>"
    
    def describe(self) -> str:
        """Return description of what this task does."""
        return "Prints context data and trace information to console for debugging. Pass-through task that doesn't modify context."
    
    def describe_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter descriptions for debug configuration."""
        return {
            "context": {
                "description": "Show context data",
                "type": "bool",
                "default": True,
                "example": "true"
            },
            "trace": {
                "description": "Show trace information",
                "type": "bool",
                "default": False,
                "example": "true"
            },
            "types": {
                "description": "Comma-separated list of data types to show (or 'all')",
                "type": "str",
                "default": "all",
                "example": "text,image"
            },
            "label": {
                "description": "Optional label to identify this debug point",
                "type": "str",
                "example": "After processing"
            }
        }
