from enum import Enum
from typing import List, Dict, Any, Optional, Deque
from dataclasses import dataclass
from collections import deque
from PIL import Image
import logging

from utils.image_utils import load_image_from_url

from .history_format import HistoryFormat
from .history_format_base import HistoryFormatBase
from .history_format_factory import HistoryFormatFactory


logger = logging.getLogger(__name__)

class HistoryFormat(Enum):
    """Available context formatting options."""
    XML = "xml"
    MINIMAL = "minimal"

class History:
    """History-related operations."""
    
    def __init__(self, 
                 max_pairs: int = 10, 
                 max_images: int = 1,
                 history_format: HistoryFormat = HistoryFormat.XML, **kwargs):
        """
        Initialize the context facade.
        
        Args:
            max_pairs: Maximum number of conversation pairs to store
            max_images: Maximum number of images to keep
            history_format: Initial context format to use
        """
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        if max_images != 1:
            raise ValueError("max_images must be 1")
            
        self.max_pairs = max_pairs
        self.max_images = max_images
        self.history_format = history_format
        
        # Initialize formatter using factory
        self._formatter = HistoryFormatFactory.create_formatter(history_format, kwargs)
        
        self._pairs: Deque[tuple[str,str]] = deque(maxlen=max_pairs)
        self._current_image: Optional[Image.Image] = None
        self._image_count = 0

    @property
    def current_image(self) -> Optional[Image.Image]:
        """Get the currently loaded image."""
        return self._current_image

    def add_image(self, image: Image.Image) -> None:
        if image:
            self._current_image = image
            logger.info(f"Successfully loaded image")


    def add_conversation_pair(self, request_text: str, response_text: str, 
            request_metadata: Dict = None) -> None:
        """Add a conversation pair to the context."""
        self._pairs.append((request_text, response_text))

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self._pairs.clear()
        self._current_image = None

    def get_stats(self) -> Dict[str, Any]:
        """Get context buffer statistics."""
        stats = {}
        stats["pairs"] = f"{len(self._pairs)}"
        stats["images"] = f"{self._image_count}"
        stats["format"] = self._formatter.format_name
        stats["max_pairs"] = f"{self.max_pairs}"
        stats["max_images"] = f"{self.max_images}"
        return stats
    
    def __str__(self)-> str:
        if not self._pairs:
            return "No conversation history available."
        
        val = "\n=== Conversation History ==="
        for i, pair in enumerate(self._pairs, 1):
            val += f"\nTurn {i}:"
            val += f"  User: {pair['request']}"
            if 'response' in pair:
                val += f"  Assistant: {pair['response']}"
            else:
                val += "  Assistant: [No response yet]"
        val += "=== End History ===\n"
        return val
    
    def set_format(self, format_type: HistoryFormat) -> None:
        """Change the formatting strategy."""
        self._formatter = HistoryFormatFactory.create_formatter(format_type)
    
    def get_formatted_history(self) -> str:
        """Get conversation context in current format."""
        if not self._pairs:
            return ""
        self._formatter._pairs = self._pairs
        return self._formatter.format_context()
    
    def clear(self) -> None:
        """Clear conversation history."""
        self._pairs.clear()
        self._image_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get context buffer statistics."""
        return {
            "pairs": f"{len(self._pairs)}/{self.max_pairs}",
            "images": f"{self._image_count}/{self.max_images}",
            "format": self._formatter.format_name
        }
    
    def get_history_description(self) -> List[Dict[str, str]]:
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