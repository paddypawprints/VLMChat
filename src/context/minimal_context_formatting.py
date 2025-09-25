from typing import List

from .base_context_format import BaseContextFormatting

class MinimalContextFormatting(BaseContextFormatting):
    """Minimal context formatting implementation."""
    
    def __init__(self, word_limit: int = 15):
        self.word_limit = word_limit
    
    @property
    def format_name(self) -> str:
        """Get the name of this formatting strategy."""
        return "minimal"
    
    def format_turn(self, request: str, response: str = None) -> str:
        """Format a conversation turn in minimal style."""
        req = self._truncate_text(request)
        if response:
            resp = self._truncate_text(response)
            return f"U: {req} | A: {resp}"
        return f"U: {req}"
    
    def format_context(self) -> str:
        """Format the entire conversation context in minimal style."""
        if not hasattr(self, '_pairs') or not self._pairs:
            return ""
            
        context_lines = []
        for pair in self._pairs:
            context_lines.append(
                self.format_turn(
                    pair.request.text,
                    pair.response.text if pair.response else None
                )
            )
        return "\n".join(context_lines)
    
    def _truncate_text(self, text: str) -> str:
        """Helper to truncate text to specified number of words."""
        words = text.split()
        if len(words) <= self.word_limit:
            return text
        return ' '.join(words[:self.word_limit]) + '...'

