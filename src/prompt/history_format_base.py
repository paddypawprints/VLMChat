"""
Base class for conversation history formatting strategies.

This module defines the abstract base class that all history formatting
implementations must inherit from, ensuring a consistent interface for
different formatting strategies.
"""

from abc import ABC, abstractmethod
from typing import Tuple
from collections import deque


class HistoryFormatBase(ABC):
    """
    Abstract base class for conversation history formatters.

    This class defines the interface that all history formatting strategies
    must implement, providing a consistent way to format conversation history
    for different use cases (XML, minimal, etc.).
    """

    @property
    @abstractmethod
    def format_name(self) -> str:
        """
        Get the name of this formatting strategy.

        Returns:
            str: Human-readable name of the format
        """
        pass

    @abstractmethod
    def format_turn(self, request: str, response: str = None) -> str:
        """
        Format a single conversation turn.

        Args:
            request: The user's request/question
            response: The assistant's response (optional)

        Returns:
            str: Formatted representation of the conversation turn
        """
        pass

    @abstractmethod
    def format_history(self, pairs: deque[Tuple[str, str]]) -> str:
        """
        Format the entire conversation history into a string.

        Args:
            pairs: Deque of conversation pairs (request, response)

        Returns:
            str: Formatted conversation history suitable for model input
        """
        pass
    
