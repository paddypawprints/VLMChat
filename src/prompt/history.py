"""
Conversation history management.

This module provides the History class for managing conversation history
with support for multiple formatting strategies, image handling, and
configurable limits on conversation pairs and images.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Deque
from dataclasses import dataclass
from collections import deque
from PIL import Image
import logging

from utils.image_utils import load_image_from_url

from .history_format_base import HistoryFormatBase
from .history_format_factory import HistoryFormatFactory


logger = logging.getLogger(__name__)

class HistoryFormat(Enum):
    """
    Available conversation history formatting options.

    Defines the different ways conversation history can be formatted
    for inclusion in prompts sent to the language model.
    """
    XML = "xml"        # XML-structured format with tags
    MINIMAL = "minimal"  # Condensed format with abbreviations

class History:
    """
    Manages conversation history with configurable formatting.

    This class handles storage and formatting of conversation history,
    supporting multiple output formats and managing conversation pairs
    and images within configured limits.
    """

    def __init__(self,
                 max_pairs: int = 10,
                 max_images: int = 1,
                 history_format: HistoryFormat = HistoryFormat.XML, **kwargs):
        """
        Initialize conversation history manager.

        Args:
            max_pairs: Maximum number of conversation pairs to retain
            max_images: Maximum number of images to keep (currently limited to 1)
            history_format: Format strategy for conversation history output
            **kwargs: Additional parameters passed to formatter

        Raises:
            ValueError: If max_pairs is not positive or max_images is not 1
        """
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        if max_images != 1:
            raise ValueError("max_images must be 1 (multiple image support not implemented)")

        self._max_pairs = max_pairs
        self._max_images = max_images
        self._history_format = history_format

        # Initialize formatter using factory pattern
        self._formatter = HistoryFormatFactory.create_formatter(history_format, kwargs)

        # Storage for conversation pairs and images
        self._pairs: Deque[tuple[str, str]] = deque(maxlen=max_pairs)
        self._current_image: Optional[Image.Image] = None
        self._image_count = 0

    @property
    def max_pairs(self) -> int:
        """Get the maximum number of conversation pairs."""
        return self._max_pairs

    @property
    def max_images(self) -> int:
        """Get the maximum number of images."""
        return self._max_images

    @property
    def current_image(self) -> Optional[Image.Image]:
        """
        Get the currently loaded image.

        Returns:
            Optional[Image.Image]: Current image if available, None otherwise
        """
        return self._current_image

    def set_current_image(self, image: Image.Image) -> None:
        """
        Set the current image for the conversation.

        Args:
            image: PIL Image object to set as current image
        """
        if image:
            self._current_image = image
            self._image_count = 1
            logger.info("Successfully set current image")

    def add_image(self, image: Image.Image) -> None:
        """
        Add an image to the conversation context.

        Args:
            image: PIL Image object to add
        """
        self.set_current_image(image)


    def add_conversation_pair(self, request_text: str, response_text: str,
                            request_metadata: Dict = None) -> None:
        """
        Add a conversation pair to the history.

        Args:
            request_text: The user's request/question
            response_text: The assistant's response
            request_metadata: Optional metadata for the request (unused currently)
        """
        self._pairs.append((request_text, response_text))

    def clear_history(self) -> None:
        """
        Clear all conversation history and reset state.

        Removes all conversation pairs and clears the current image.
        """
        self._pairs.clear()
        self._current_image = None
        self._image_count = 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation history statistics.

        Returns:
            Dict[str, Any]: Dictionary containing current statistics about
                           conversation pairs, images, and configuration
        """
        return {
            "pairs": f"{len(self._pairs)}/{self._max_pairs}",
            "images": f"{self._image_count}/{self._max_images}",
            "format": self._formatter.format_name
        }
    
    def __str__(self) -> str:
        """
        Get a human-readable string representation of the conversation history.

        Returns:
            str: Formatted conversation history for display
        """
        if not self._pairs:
            return "No conversation history available."

        lines = ["\n=== Conversation History ==="]
        for i, (request, response) in enumerate(self._pairs, 1):
            lines.append(f"Turn {i}:")
            lines.append(f"  User: {request}")
            lines.append(f"  Assistant: {response}" if response else "  Assistant: [No response yet]")
        lines.append("=== End History ===\n")
        return "\n".join(lines)
    
    def set_format(self, format_type: HistoryFormat) -> None:
        """
        Change the conversation history formatting strategy.

        Args:
            format_type: New format type to use for history output
        """
        self._formatter = HistoryFormatFactory.create_formatter(format_type)
        self._history_format = format_type
    
    def get_formatted_history(self) -> str:
        """
        Get formatted conversation history using current formatter.

        Returns:
            str: Conversation history formatted according to current format strategy
        """
        if not self._pairs:
            return ""
        # Pass pairs to formatter and get formatted output
        return self._formatter.format_history(self._pairs)
    
    
    def get_history_description(self) -> List[Dict[str, str]]:
        """
        Get conversation history as a list of text-only dictionaries.

        Returns:
            List[Dict[str, str]]: List of conversation entries with 'request'
                                 and optional 'response' keys
        """
        context = []
        for request, response in self._pairs:
            entry = {'request': request}
            if response:
                entry['response'] = response
            context.append(entry)

        return context