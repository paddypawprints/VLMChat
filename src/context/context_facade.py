from typing import List, Dict, Any, Optional, Tuple
from PIL import Image
import logging
from utils.image_utils import get_pil_image_from_url

from .llm_context_buffer import LLMContextBuffer
from .context_format import ContextFormat
from .message_formatter import MessageFormatter
from .metadata_retriever import MetadataRetriever
from .message_structure import MessageStructureManager

logger = logging.getLogger(__name__)

class ContextFacade:
    """Facade for all context-related operations."""
    
    def __init__(self, 
                 max_pairs: int = 10, 
                 max_images: int = 2,
                 context_format: ContextFormat = ContextFormat.XML):
        """
        Initialize the context facade.
        
        Args:
            max_pairs: Maximum number of conversation pairs to store
            max_images: Maximum number of images to keep
            context_format: Initial context format to use
        """
        self._context_buffer = LLMContextBuffer(
            max_pairs=max_pairs,
            max_images=max_images,
            enable_metadata=True
        )
        self._message_formatter = MessageFormatter(self)
        self._metadata_retriever = MetadataRetriever()
        self.set_format(context_format)
        self._current_image: Optional[Image.Image] = None
        self._message_manager = MessageStructureManager(self)

    @property
    def current_image(self) -> Optional[Image.Image]:
        """Get the currently loaded image."""
        return self._current_image

    def load_image_from_url(self, image_url: str) -> bool:
        """Load an image from URL."""
        image = get_pil_image_from_url(image_url)
        if image:
            self._current_image = image
            logger.info(f"Successfully loaded image from: {image_url}")
            return True
        logger.error(f"Failed to load image from: {image_url}")
        return False

    def load_image_from_file(self, image_path: str) -> bool:
        """Load an image from file path."""
        try:
            self._current_image = Image.open(image_path).convert('RGB')
            logger.info(f"Successfully loaded image from: {image_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load image from {image_path}: {e}")
            return False

    def set_image_from_camera(self, image: Image.Image) -> None:
        """Set the current image from camera capture."""
        self._current_image = image

    def add_conversation_pair(self, request_text: str, response_text: str, 
                            request_metadata: Dict = None) -> None:
        """Add a conversation pair to the context."""
        self._context_buffer.add_pair(
            request_text=request_text,
            response_text=response_text,
            request_metadata=request_metadata
        )

    def get_formatted_context(self) -> str:
        """Get conversation context in current format."""
        return self._context_buffer.get_formatted_context()

    def get_text_only_context(self) -> List[Dict[str, str]]:
        """Get text-only version of context."""
        return self._context_buffer.get_text_only_context()

    def set_format(self, format_type: ContextFormat) -> None:
        """Change the context format."""
        self._context_buffer.set_context_format(format_type)

    def clear_history(self) -> None:
        """Clear all conversation history."""
        self._context_buffer.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get context buffer statistics."""
        return self._context_buffer.get_stats()

    def show_conversation_history(self) -> None:
        """Display the current conversation history."""
        context = self.get_text_only_context()
        
        if not context:
            print("No conversation history available.")
            return
        
        print("\n=== Conversation History ===")
        for i, pair in enumerate(context, 1):
            print(f"\nTurn {i}:")
            print(f"  User: {pair['request']}")
            if 'response' in pair:
                print(f"  Assistant: {pair['response']}")
            else:
                print("  Assistant: [No response yet]")
        print("=== End History ===\n")

    def create_model_messages(self, user_input: str) -> List[Dict[str, Any]]:
        """Create messages for model input, including metadata retrieval."""
        retrieved_metadata = self._message_manager.retrieve_metadata(user_input)
        
        return self._message_manager.create_messages(
            user_input=user_input,
            retrieved_metadata=retrieved_metadata
        )