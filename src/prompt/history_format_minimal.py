"""
Minimal conversation history formatter.

This module implements a compact formatting strategy for conversation history,
using abbreviated labels and truncated text to minimize token usage while
maintaining readability.
"""

from typing import Tuple
from collections import deque
from .history_format_base import HistoryFormatBase


class HistoryFormatMinimal(HistoryFormatBase):
    """
    Minimal conversation history formatter.

    Provides a compact formatting strategy that uses abbreviated labels
    (U: for user, A: for assistant) and optionally truncates long text
    to minimize token usage in model inputs.
    """

    def __init__(self, **kwargs):
        """
        Initialize minimal formatter with optional configuration.

        Args:
            **kwargs: Configuration options including:
                     word_limit (int): Maximum words per text segment (default: 15)
        """
        self._word_limit = kwargs.get("word_limit", 15)

    @property
    def format_name(self) -> str:
        """
        Get the name of this formatting strategy.

        Returns:
            str: The name "minimal" identifying this formatter
        """
        return "minimal"

    @property
    def word_limit(self) -> int:
        """
        Get the word limit for text truncation.

        Returns:
            int: Maximum number of words per text segment
        """
        return self._word_limit

    def format_turn(self, request: str, response: str = None) -> str:
        """
        Format a single conversation turn in minimal style.

        Uses "U:" prefix for user messages and "A:" for assistant responses,
        with optional text truncation to stay within word limits.

        Args:
            request: The user's request/question
            response: The assistant's response (optional)

        Returns:
            str: Minimally formatted conversation turn
        """
        req = self._truncate_text(request)
        if response:
            resp = self._truncate_text(response)
            return f"U: {req} | A: {resp}"
        return f"U: {req}"

    def format_history(self, pairs: deque[Tuple[str, str]]) -> str:
        """
        Format the entire conversation history in minimal style.

        Creates a compact representation of the conversation with each
        turn on a separate line using abbreviated labels.

        Args:
            pairs: Deque of conversation pairs (request, response)

        Returns:
            str: Minimally formatted conversation history
        """
        if not pairs:
            return ""

        context_lines = []
        for request, response in pairs:
            context_lines.append(self.format_turn(request, response))
        return "\n".join(context_lines)

    def _truncate_text(self, text: str) -> str:
        """
        Truncate text to the specified word limit.

        Args:
            text: Text to potentially truncate

        Returns:
            str: Original text or truncated version with ellipsis
        """
        words = text.split()
        if len(words) <= self._word_limit:
            return text
        return ' '.join(words[:self._word_limit]) + '...'

