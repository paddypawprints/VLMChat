"""
History update task for pipeline integration.

Manages conversation history with separate prompt and response operations.
Maintains state across multiple invocations within a pipeline.
"""

import logging
from typing import Dict, Any, Optional
from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...prompt.history import History

logger = logging.getLogger(__name__)


@register_task('history')
class HistoryUpdateTask(BaseTask):
    """
    Pipeline task that manages conversation history with automatic mode detection.
    
    Automatically detects operation mode based on History state:
    - If history is empty or last pair is complete → PROMPT mode: process new user input
    - If last pair has empty response → RESPONSE mode: capture model output
    
    Use with id parameter to share History state across multiple task instances.
       
    Example DSL:
        history_update(id="hist", format="xml") -> smolvlm() -> history_update(id="hist")
    """
    
    def __init__(self, history: History = None, task_id: str = "history_update"):
        """
        Initialize history update task.
        
        Args:
            history: History manager instance to update (optional, can be configured)
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.history = history
        
        # Define contracts
        # Input: Requires TEXT (at least one entry)
        # Output: Produces TEXT (modified in prompt mode, unchanged in response mode)
        self.input_contract = {
            ContextDataType.TEXT: str
        }
        self.output_contract = {
            ContextDataType.TEXT: str
        }
    
    def configure(self, **params) -> None:
        """
        Configure history update from parameters (DSL support).
        
        Args:
            **params: Keyword arguments with configuration
                - format: History format type ("markdown", "json", "simple", "xml")
                - max_pairs: Maximum conversation pairs to retain (default: 10)
        
        Example:
            task.configure(format="xml", max_pairs=20)
        """
        # Create history if not provided (first invocation only)
        if self.history is None:
            max_pairs = params.get("max_pairs", 10)
            from ...prompt.history_format import HistoryFormat
            format_str = params.get("format", "xml").upper()
            format_type = HistoryFormat[format_str] if format_str in HistoryFormat.__members__ else HistoryFormat.XML
            self.history = History(max_pairs=max_pairs, history_format=format_type)
        
        # Update format if provided in subsequent calls
        elif "format" in params:
            from ...prompt.history_format import HistoryFormat
            format_str = params["format"].upper()
            format_type = HistoryFormat[format_str] if format_str in HistoryFormat.__members__ else HistoryFormat.XML
            self.history.set_format(format_type)
    
    def run(self, context: Context) -> Context:
        """
        Process conversation history with automatic mode detection.
        
        Detects mode based on History state:
        - Empty history or last pair complete → PROMPT mode (new user input)
        - Last pair incomplete (empty response) → RESPONSE mode (capture model output)
        
        Args:
            context: Pipeline context containing TEXT with at least 1 entry
            
        Returns:
            Context with TEXT modified (prompt mode) or unchanged (response mode)
        """
        # Ensure history is initialized
        if self.history is None:
            self.history = History()
        
        # Get TEXT entries from context
        if ContextDataType.TEXT not in context.data or not context.data[ContextDataType.TEXT]:
            raise ValueError(f"Task {self.task_id}: TEXT not found in context")
        
        text_list = context.data[ContextDataType.TEXT]
        
        # Auto-detect mode based on History state
        is_response_mode = (
            self.history._pairs and  # History has pairs
            not self.history._pairs[-1][1]  # Last pair has empty response
        )
        
        logger.debug(f"Task {self.task_id}: text_list={text_list}, history._pairs={list(self.history._pairs)}, is_response_mode={is_response_mode}")
        
        if is_response_mode:
            # RESPONSE MODE: Capture model output and update last history pair
            if not text_list:
                raise ValueError(f"Task {self.task_id}: No TEXT entry to process as response")
            
            # Read (but don't remove) the model's output
            model_output = text_list[-1]
            
            # Update the most recent history entry with the response
            request, _ = self.history._pairs[-1]
            self.history._pairs[-1] = (request, model_output)
            
        else:
            # PROMPT MODE: Process new user input
            if not text_list:
                raise ValueError(f"Task {self.task_id}: No TEXT entry to process as prompt")
            
            # Remove and save the user's raw input
            user_input = text_list.pop()
            
            # Generate formatted prompt including conversation history
            formatted_history = self.history.get_formatted_history()
            
            # Always append formatted history (even if empty string)
            # This ensures TEXT list has 2 items: [history, user_input]
            text_list.append(formatted_history if formatted_history else "")
            
            # Add user input back to TEXT list
            text_list.append(user_input)
            
            # Store user input in history as Request (response will come later)
            self.history.add_conversation_pair(
                request_text=user_input,
                response_text=""  # Empty for now, will be filled by response mode
            )
        
        return context
    
    def describe(self) -> str:
        """Return description of what this task does."""
        return ("Manages conversation history with automatic mode detection. "
                "Detects PROMPT mode (new user input) when history is empty or last pair is complete. "
                "Detects RESPONSE mode (capture output) when last pair has empty response. "
                "PROMPT mode: removes last TEXT, appends formatted history + user input (2 items). "
                "RESPONSE mode: reads last TEXT, stores in history. "
                "Use 'id' parameter in DSL to share history state across task instances.")
    
    def describe_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter descriptions for history update configuration."""
        return {
            "format": {
                "description": "History format type",
                "type": "str",
                "default": "xml",
                "example": "simple",
                "options": ["markdown", "json", "simple", "xml"]
            },
            "max_pairs": {
                "description": "Maximum conversation pairs to retain",
                "type": "int",
                "default": 10,
                "example": "20"
            }
        }
