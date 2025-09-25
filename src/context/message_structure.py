from typing import List, Dict, Any, Optional, Deque
from dataclasses import dataclass
from collections import deque
from PIL import Image
import logging
from services.rag_service import RAGService

from .context_format import ContextFormat
from .format_factory import ContextFormatFactory

logger = logging.getLogger(__name__)

@dataclass
class MessageContent:
    text: str
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class ConversationPair:
    request: MessageContent
    response: Optional[MessageContent] = None
    image: Optional[Image.Image] = None

class MessageStructureManager:
    """Manages message structures, conversation history, and metadata retrieval."""
    
    def __init__(self, 
                 max_pairs: int = 50,
                 max_images: int = 100,
                 enable_metadata: bool = True,
                 context_format: ContextFormat = ContextFormat.XML):
        """Initialize message structure manager."""
        if max_pairs <= 0:
            raise ValueError("max_pairs must be positive")
        if max_images < 0:
            raise ValueError("max_images must be non-negative")
            
        self.max_pairs = max_pairs
        self.max_images = max_images
        self.enable_metadata = enable_metadata
        
        # Initialize formatter using factory
        self._formatter = ContextFormatFactory.create_formatter(context_format)
        
        self._pairs: Deque[ConversationPair] = deque(maxlen=max_pairs)
        self._current_image: Optional[Image.Image] = None
        self._image_count = 0
        
        # Initialize RAG service for metadata retrieval
        self._rag_service = RAGService()
    
    def retrieve_metadata(self, user_input: str) -> str:
        """
        Retrieve metadata for the current context.
        
        Args:
            user_input: User's text input
            
        Returns:
            Retrieved metadata as string
        """
        if not self._current_image:
            return ""
            
        context_history = [
            {"request": pair.request.text, "response": pair.response.text if pair.response else ""}
            for pair in self._pairs
        ]
        
        return self._rag_service.retrieve_metadata(
            image=self._current_image,
            context=user_input,
            conversation_history=context_history
        )

    def set_format(self, format_type: ContextFormat) -> None:
        """Change the formatting strategy."""
        self._formatter = ContextFormatFactory.create_formatter(format_type)
    
    def get_formatted_context(self) -> str:
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