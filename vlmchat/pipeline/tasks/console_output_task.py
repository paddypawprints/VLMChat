"""
Console output task for pipeline integration.

Displays various context data types to console with appropriate formatting.
"""

import logging
import numpy as np
from typing import Dict, Any, List
from ..task_base import BaseTask, Context, ContextDataType, register_task

logger = logging.getLogger(__name__)


@register_task('output')
class ConsoleOutputTask(BaseTask):
    """
    Pipeline sink task that displays context data to console.
    
    Can display different context types with appropriate formatting:
    - TEXT: Plain text output (default)
    - SIMILARITY_SCORES: Ranked similarity matrix
    - DETECTIONS: Detection summary
    - EMBEDDINGS: Embedding shape info
    
    Examples:
        output()                          # Display TEXT (default)
        output(types="text")              # Explicit TEXT
        output(types="similarities")      # Display SIMILARITY_SCORES
        output(types="detections")        # Display DETECTIONS summary
        output(types="text,similarities") # Display multiple types
    """
    
    def __init__(self, task_id: str = "console_output"):
        """
        Initialize console output task.
        
        Args:
            task_id: Unique identifier for this task
        """
        super().__init__(task_id)
        self.which = "all"  # "all", "first", or "last" (for TEXT)
        self.output_types = ["text"]  # Default: display TEXT
        self.top_k = None  # For similarities: show top K matches per text
        
        # Define contracts
        self.input_contract = {}  # Flexible: accepts various types
        self.output_contract = {}  # Sink task produces no output
    
    def configure(self, **params) -> None:
        """
        Configure console output from parameters (DSL support).
        
        Args:
            **params: Keyword arguments with configuration
                - types: Comma-separated list of types to display: "text", "similarities", "detections", "embeddings"
                - which: Which TEXT entries to print: "all", "first", or "last" (default: "all")
                - top_k: For similarities, show top K matches per text (default: None = all)
        
        Example:
            task.configure(types="text")
            task.configure(types="similarities", top_k="5")
            task.configure(types="text,similarities")
        """
        if "types" in params:
            types_str = params["types"].lower()
            self.output_types = [t.strip() for t in types_str.split(",")]
        
        if "which" in params:
            which_val = params["which"].lower()
            if which_val in ("all", "first", "last"):
                self.which = which_val
            else:
                raise ValueError(f"Invalid 'which' parameter: {params['which']}. Must be 'all', 'first', or 'last'")
        
        if "top_k" in params:
            self.top_k = int(params["top_k"])
    
    
    def _format_text(self, context: Context) -> List[str]:
        """Format TEXT for display."""
        if ContextDataType.TEXT not in context.data or not context.data[ContextDataType.TEXT]:
            return []
        
        text_list = context.data[ContextDataType.TEXT]
        
        # Determine which entries to output
        if self.which == "first":
            return [text_list[0]]
        elif self.which == "last":
            return [text_list[-1]]
        else:  # "all"
            return text_list
    
    def _format_similarities(self, context: Context) -> List[str]:
        """Format SIMILARITY_SCORES for display."""
        if ContextDataType.SIMILARITY_SCORES not in context.data:
            return ["No similarity scores in context"]
        
        scores = context.data[ContextDataType.SIMILARITY_SCORES]
        matrix = scores.get('matrix', np.array([]))
        texts = scores.get('texts', [])
        detection_ids = scores.get('detection_ids', [])
        
        if matrix.size == 0:
            return ["Empty similarity matrix"]
        
        output = []
        output.append("\n=== Similarity Scores ===")
        
        for i, text in enumerate(texts):
            scores_for_text = matrix[i, :]
            # Sort by similarity (descending)
            sorted_indices = np.argsort(scores_for_text)[::-1]
            
            output.append(f"\n'{text}':")
            
            # Apply top_k filter if specified
            indices_to_show = sorted_indices[:self.top_k] if self.top_k else sorted_indices
            
            for j in indices_to_show:
                output.append(f"  {detection_ids[j]}: {scores_for_text[j]:.3f}")
        
        output.append("")
        return output
    
    def _format_detections(self, context: Context) -> List[str]:
        """Format DETECTIONS summary for display."""
        if ContextDataType.DETECTIONS not in context.data:
            return ["No detections in context"]
        
        detections = context.data[ContextDataType.DETECTIONS]
        
        if not detections:
            return ["No detections found"]
        
        output = []
        output.append(f"\n=== Detections ({len(detections)}) ===")
        
        for i, det in enumerate(detections):
            # Get detection info
            category = getattr(det, 'object_category', 'unknown')
            conf = getattr(det, 'conf', 0.0)
            box = getattr(det, 'box', (0, 0, 0, 0))
            
            output.append(f"[{i}] {category} ({conf:.2f}) at {box}")
        
        output.append("")
        return output
    
    def _format_embeddings(self, context: Context) -> List[str]:
        """Format EMBEDDINGS summary for display."""
        if ContextDataType.EMBEDDINGS not in context.data:
            return ["No embeddings in context"]
        
        embeddings = context.data[ContextDataType.EMBEDDINGS]
        
        if not embeddings:
            return ["No embeddings found"]
        
        output = []
        
        # Check if nested (from merge) or flat
        if isinstance(embeddings[0], list) and len(embeddings[0]) == 2 and isinstance(embeddings[0][0], str):
            # Flat list of [label, embedding] pairs
            output.append(f"\n=== Embeddings ({len(embeddings)}) ===")
            for label, emb in embeddings:
                output.append(f"  '{label}': shape={emb.shape}, dtype={emb.dtype}")
        else:
            # Nested list (from merge)
            output.append(f"\n=== Embeddings ({len(embeddings)} groups) ===")
            for i, group in enumerate(embeddings):
                output.append(f"Group {i}: {len(group)} embeddings")
                for label, emb in group[:3]:  # Show first 3
                    output.append(f"  '{label}': shape={emb.shape}")
                if len(group) > 3:
                    output.append(f"  ... and {len(group) - 3} more")
        
        output.append("")
        return output
    
    def run(self, context: Context) -> Context:
        """
        Display context data to console via pipeline runner's output queue.
        
        Args:
            context: Pipeline context containing data to display
            
        Returns:
            Unchanged context (sink task)
        """
        runner = getattr(context, 'pipeline_runner', None)
        if runner is None:
            logger.error(f"Task {self.task_id}: pipeline_runner is None - context attrs: {dir(context)}")
            raise RuntimeError(f"Task {self.task_id}: context.pipeline_runner not set - cannot send output")
        
        # Map type names to formatters
        formatters = {
            'text': self._format_text,
            'similarities': self._format_similarities,
            'detections': self._format_detections,
            'embeddings': self._format_embeddings
        }
        
        # Format and send output for each requested type
        for output_type in self.output_types:
            formatter = formatters.get(output_type)
            if formatter:
                lines = formatter(context)
                for line in lines:
                    runner.output_queue.put(line)
            else:
                runner.output_queue.put(f"Unknown output type: {output_type}")
        
        # Return context unchanged
        return context
    
    def describe(self) -> str:
        """Return description of what this task does."""
        return ("Displays context data to console with appropriate formatting. "
                "Supports TEXT, SIMILARITY_SCORES, DETECTIONS, and EMBEDDINGS.")
    
    def describe_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Return parameter descriptions for console output configuration."""
        return {
            "types": {
                "description": "Comma-separated list of data types to display",
                "type": "str",
                "choices": ["text", "similarities", "detections", "embeddings"],
                "default": "text",
                "example": "text,similarities"
            },
            "which": {
                "description": "Which TEXT entries to print (when types includes 'text')",
                "type": "str",
                "choices": ["all", "first", "last"],
                "default": "all",
                "example": "last"
            },
            "top_k": {
                "description": "Show top K matches per text (for similarities)",
                "type": "int",
                "default": "None (all)",
                "example": "5"
            }
        }
