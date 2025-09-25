from abc import ABC, abstractmethod
from typing import Tuple
from collections import deque

class HistoryFormatBase(ABC):
    """Base class defining interface for context formatting strategies."""
    
    @property
    @abstractmethod
    def format_name(self) -> str:
        """Name of the formatting strategy."""
        pass

    @abstractmethod
    def format_turn(request: str, response: str = None) -> str:
        """Format a single conversation turn."""
        pass
    
    @abstractmethod
    def format_history(self, pairs: deque[Tuple[str,str]]) -> str:
        """Format the entire conversation context into a string."""
        pass
    
