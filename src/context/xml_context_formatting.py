from typing import List

from .base_context_format import BaseContextFormatting

class XMLContextFormatting(BaseContextFormatting):
    """XML-based context formatting implementation."""
    
    @property
    def format_name(self) -> str:
        """Get the name of this formatting strategy."""
        return "xml"
    
    def format_turn(self, request: str, response: str = None) -> str:
        """Format a conversation turn with XML tags."""
        parts = []
        parts.append(f"<user>{request}</user>")
        if response:
            parts.append(f"<assistant>{response}</assistant>")
        return "\n".join(parts)
    
    def format_context(self) -> str:
        """
        Get formatted context suitable for inclusion in system prompts.
        
        Returns:
            Formatted conversation history as a string with XML-style tags
        """
        if not hasattr(self, '_pairs') or not self._pairs:
            return ""
        
        context_parts = ["<conversation_history>"]
        
        for i, pair in enumerate(self._pairs):
            context_parts.append(f"<turn_{i + 1}>")
            context_parts.append(f"<user>{pair.request.text}</user>")
            
            if pair.response:
                context_parts.append(f"<assistant>{pair.response.text}</assistant>")
            
            context_parts.append(f"</turn_{i + 1}>")
        
        context_parts.append("</conversation_history>")
        return "\n".join(context_parts)


