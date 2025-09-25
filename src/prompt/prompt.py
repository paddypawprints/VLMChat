from typing import List, Dict, Any, Optional
from enum import Enum
from PIL import Image
import logging
from src.prompt.history import History, HistoryFormat

logger = logging.getLogger(__name__)

class Prompt:
    """Facade for all context-related operations."""

    def __init__(self, **kwargs):
        self._history = History(
            max_pairs=kwargs.get("max_pairs", 10),
            max_images=kwargs.get("max_images", 1),
            history_format=kwargs.get("history_format", HistoryFormat.XML),
            **kwargs
        )
        self._current_image: Optional[Image.Image] = None
        self._user_input: Optional[str] = None

    @property
    def current_image(self) -> Optional[Image.Image]:
        """Get the currently loaded image."""
        return self._current_image

    def add_assistant_response(self, user_input, assistant_response: str) -> None:
        """Add a conversation pair to the context."""
        self._history.add_pair(request_text=user_input, response_text=assistant_response)
    
    @property
    def user_input(self) -> Optional[str]:
        """Get the current user input."""
        return self._user_input

    @property
    def history(self) -> History:
        """Get text-only version of context."""
        return self._history

    def get_stats(self) -> Dict[str, Any]:
        """Get context buffer statistics."""
        return self._history.get_stats()

