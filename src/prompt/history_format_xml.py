from typing import List, Tuple
from collections import deque
from .history_format_base import HistoryFormatBase

class HistoryFormatXML(HistoryFormatBase):
    """XML-based context formatting implementation."""
    
    @property
    def format_name(self) -> str:
        """Get the name of this formatting strategy."""
        return "xml"
    
    def format_turn(self, request: str, response: str = None) -> str:
        """Format a conversation turn with XML tags."""
        parts = f"<user>{request}</user>"
        if response:
            parts += f"<assistant>{response}</assistant>"
        return parts

    def format_context(self, pairs: deque[Tuple[str,str]]) -> str:
        """
        Get formatted context suitable for inclusion in system prompts.
        
        Returns:
            Formatted conversation history as a string with XML-style tags
        """
        if not pairs:
            return ""
        
        context_parts = "<conversation_history>"
        
        for i, pair in enumerate(pairs):
            context_parts += f"<turn_{i + 1}>"
            context_parts += f"<user>{pair.request}</user>"
            
            if pair.response:
                context_parts += f"<assistant>{pair.response}</assistant>"

            context_parts += f"</turn_{i + 1}>"

        context_parts += "</conversation_history>"
        return context_parts
