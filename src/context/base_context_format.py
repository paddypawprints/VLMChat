from abc import ABC, abstractmethod
from typing import List, Dict
from .context_format import ContextFormat

class BaseContextFormatting(ABC):
    """Base class defining interface for context formatting strategies."""
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Name of the formatting strategy."""
        pass

    @abstractmethod
    def format_turn(self, request: str, response: str = None) -> str:
        """Format a single conversation turn."""
        pass
    
    @abstractmethod
    def format_context(self) -> str:
        """Format the entire conversation context into a string."""
        pass
    
    def get_text_only_context(self) -> List[Dict[str, str]]:
        """
        Get context with only text content (no images).
        
        Returns:
            List of text-only context entries
        """
        if not hasattr(self, '_pairs'):
            return []
            
        context = []
        for pair in self._pairs:
            entry = {'request': pair.request.text}
            if pair.response:
                entry['response'] = pair.response.text
            context.append(entry)
        
        return context