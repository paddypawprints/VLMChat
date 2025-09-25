from typing import Tuple
from .history_format_base import HistoryFormatBase
from collections import deque

class HistoryFormatMinimal(HistoryFormatBase):
    """Minimal context formatting implementation."""
    
    def __init__(self, **kwargs):
        self.word_limit = kwargs.get("word_limit", 15)
    
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
    
    def format_history(self, pairs: deque[Tuple[str,str]]) -> str:
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

