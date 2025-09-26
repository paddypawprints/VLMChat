"""
Prompt management and conversation facade.

This module provides the Prompt class which serves as a facade for managing
conversation prompts, history, and context. It integrates with the History
class to provide a unified interface for prompt-related operations.
"""

from typing import Dict, Any, Optional
from PIL import Image
import logging
from prompt.history import History
from prompt.history_format import HistoryFormat

logger = logging.getLogger(__name__)

class Prompt:
    """
    Facade for all prompt and context-related operations.

    This class provides a unified interface for managing conversation prompts,
    history, and context. It acts as a facade over the History class while
    maintaining additional prompt-specific state.
    """

    def __init__(self, **kwargs):
        """
        Initialize the prompt facade with history management.

        Args:
            **kwargs: Configuration parameters passed to History initialization
                     including max_pairs, max_images, and history_format
        """
        self._history = History(
            # max_pairs=kwargs.get("max_pairs", 10),
            # max_images=kwargs.get("max_images", 1),
            # history_format=kwargs.get("history_format", HistoryFormat.XML),
            **kwargs
        )
        self._current_image: Optional[Image.Image] = None
        self._user_input: Optional[str] = None

    @property
    def current_image(self) -> Optional[Image.Image]:
        """
        Get the currently loaded image.

        Returns:
            Optional[Image.Image]: Current image if available, None otherwise
        """
        return self._current_image
    
    @current_image.setter
    def current_image(self, new_image: Optional[Image.Image]):
        """Set the currently loaded image."""
        # You can add validation logic here if needed
        if new_image is not None and not isinstance(new_image, Image.Image):
            raise TypeError("The provided value is not a valid Image object.")
        self._current_image = new_image

    @property
    def user_input(self) -> Optional[str]:
        """
        Get the current user input text.

        Returns:
            Optional[str]: Current user input if available, None otherwise
        """
        return self._user_input

    @property
    def history(self) -> History:
        """
        Get the conversation history manager.

        Returns:
            History: The history management instance
        """
        return self._history

    def add_assistant_response(self, user_input: str, assistant_response: str) -> None:
        """
        Add a conversation pair to the history.

        Args:
            user_input: The user's input text
            assistant_response: The assistant's response text
        """
        self.history.add_conversation_pair(request_text=user_input, response_text=assistant_response)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation context statistics.

        Returns:
            Dict[str, Any]: Statistics about the conversation history
        """
        return self._history.get_stats()

