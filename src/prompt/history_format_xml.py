"""
XML-based conversation history formatter.

This module implements XML-structured formatting for conversation history,
using XML tags to clearly delineate different parts of the conversation.
"""

from typing import Tuple
from collections import deque
from .history_format_base import HistoryFormatBase


class HistoryFormatXML(HistoryFormatBase):
    """
    XML-based conversation history formatter.

    Formats conversation history using XML tags to provide clear structure
    and delineation between user requests and assistant responses.
    """

    @property
    def format_name(self) -> str:
        """
        Get the name of this formatting strategy.

        Returns:
            str: The name "xml" identifying this formatter
        """
        return "xml"

    def format_turn(self, request: str, response: str = None) -> str:
        """
        Format a single conversation turn with XML tags.

        Args:
            request: The user's request/question
            response: The assistant's response (optional)

        Returns:
            str: XML-formatted conversation turn
        """
        parts = f"<user>{request}</user>"
        if response:
            parts += f"<assistant>{response}</assistant>"
        return parts

    def format_history(self, pairs: deque[Tuple[str, str]]) -> str:
        """
        Format the entire conversation history with XML structure.

        Creates a hierarchical XML structure with conversation_history as the
        root element and numbered turns containing user/assistant pairs.

        Args:
            pairs: Deque of conversation pairs (request, response)

        Returns:
            str: XML-formatted conversation history suitable for model input
        """
        if not pairs:
            return ""

        context_parts = ["<conversation_history>"]

        for i, (request, response) in enumerate(pairs, 1):
            context_parts.append(f"<turn_{i}>")
            context_parts.append(f"<user>{request}</user>")

            if response:
                context_parts.append(f"<assistant>{response}</assistant>")

            context_parts.append(f"</turn_{i}>")

        context_parts.append("</conversation_history>")
        return "".join(context_parts)
