"""
No-history conversation formatter.

This formatter returns an empty string for any requested history formatting
operations. Useful for privacy-preserving or minimal-context runs where the
conversation history should not be included in prompts.
"""

from collections import deque
from typing import Tuple
from .history_format_base import HistoryFormatBase


class HistoryFormatNoHistory(HistoryFormatBase):
    @property
    def format_name(self) -> str:
        return "nohistory"

    def format_turn(self, request: str, response: str = None) -> str:
        # Always return empty string for any turn
        return ""

    def format_history(self, pairs: deque[Tuple[str, str]]) -> str:
        # Always return empty string for full history
        return ""
