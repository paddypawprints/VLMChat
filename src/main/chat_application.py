# main/chat_application.py
"""Main chat application orchestrating all components."""

import logging
from PIL import Image

from models.smol_vlm_model import SmolVLMModel  # Change this line
from models.model_config import ModelConfig
from services.rag_service import RAGService
from utils.image_utils import get_pil_image_from_url
from context.context_format import ContextFormat
from utils.camera import IMX500ObjectDetection
from context.context_facade import ContextFacade
from models.response_generator import ResponseGenerator

logger = logging.getLogger(__name__)

class SmolVLMChatApplication:
    """Main application class for SmolVLM chat interface."""
    
    def __init__(self, 
                 model_path: str = "HuggingFaceTB/SmolVLM2-256M-Instruct", 
                 use_onnx: bool = True, 
                 max_context_pairs: int = 10,
                 max_context_images: int = 2,
                 context_format: ContextFormat = ContextFormat.XML):
        """
        Initialize the chat application.
        
        Args:
            model_path: Path to the SmolVLM model
            use_onnx: Whether to use ONNX runtime
            max_context_pairs: Maximum number of conversation pairs to keep in context
            max_context_images: Maximum number of images to keep in context
            context_format: Format to use for conversation context (XML, minimal, or text_only)
        """
        # Setup logging
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        # Initialize components
        self.config = ModelConfig(model_path=model_path)
        self.model = SmolVLMModel(self.config, use_onnx=use_onnx)
        self.response_generator = ResponseGenerator(self.model)
        self.rag_service = RAGService()
        
        # Initialize context facade only
        self.context = ContextFacade(
            max_pairs=max_context_pairs,
            max_images=max_context_images,
            context_format=context_format
        )
        
        # Application state
        # Remove current_image from application state
        
        # Initialize camera
        self.camera = IMX500ObjectDetection()
        
        logger.info("SmolVLM Chat Application initialized successfully")

    def set_context_format(self, context_format: ContextFormat) -> None:
        """Set the context format."""
        self.context_format = context_format
        self.context.set_format(context_format)

    def capture_from_camera(self) -> bool:
        """
        Capture an image from the camera and load it into the current context.
        
        Returns:
            bool: True if capture successful, False otherwise
        """
        try:
            filepath, image = self.camera.capture_single_image()
            self.context.set_image_from_camera(image)
            print(f"Captured image saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            return False

   
    def process_query(self, user_input: str, stream_output: bool = True) -> str:
        """Process a user query with the current image and conversation context."""
        if not self.context.current_image:
            return "No image loaded. Please load an image first."
        
        logger.info("Processing user query with context")
        
        # Create messages with metadata and context
        messages = self.context.create_model_messages(
            user_input=user_input,
            current_image=self.context.current_image
        )
        
        # Generate response using the generator
        try:
            response = self.response_generator.generate_response(
                messages=messages,
                images=[self.context.current_image],
                stream_output=stream_output
            )
            
            # Add the conversation pair to context
            self.context.add_conversation_pair(
                request_text=user_input,
                response_text=response,
                request_metadata={'has_image': True, 'image_source': 'current_session'}
            )
            
            logger.info(f"Context stats: {self.context.get_stats()}")
            return response
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error generating response: {e}"
    
    def _print_help_message(self) -> None:
        """Display available commands and their descriptions."""
        print("Commands:")
        print("  /load_url <url>  - Load image from URL")
        print("  /load_file <path> - Load image from file")
        print("  /clear_context   - Clear conversation history")
        print("  /show_context    - Show current conversation history")
        print("  /context_stats   - Show context buffer statistics")
        print("  /quit - Exit the application")
        print("  /help - Show this help message")
        print("  /format <xml|minimal|text_only> - Change context format")
        print("  /camera - Capture image from camera")

    def run_interactive_chat(self):
        """Run the interactive chat loop."""
        print("=== SmolVLM Interactive Chat ===")
        self._print_help_message()
        print()
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.startswith('/quit'):
                    print("Goodbye!")
                    break
                elif user_input.startswith('/help'):
                    self._print_help_message()
                    continue
                elif user_input.startswith('/load_url '):
                    url = user_input[10:].strip()
                    if self.context.load_image_from_url(url):
                        print("Image loaded successfully!")
                        self.context.clear_history()
                        print("Conversation history cleared for new image.")
                    else:
                        print("Failed to load image.")
                    continue
                elif user_input.startswith('/load_file '):
                    path = user_input[11:].strip()
                    if self.context.load_image_from_file(path):
                        print("Image loaded successfully!")
                        self.context.clear_history()
                        print("Conversation history cleared for new image.")
                    else:
                        print("Failed to load image.")
                    continue
                elif user_input.startswith('/clear_context'):
                    self.context.clear_history()
                    print("Conversation history cleared.")
                    continue
                elif user_input.startswith('/show_context'):
                    self.context.show_conversation_history()
                    continue
                elif user_input.startswith('/context_stats'):
                    stats = self.context.get_stats()
                    print("Context Buffer Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                elif user_input.startswith("/format"):
                    try:
                        _, format_name = user_input.split(maxsplit=1)
                        new_format = ContextFormat(format_name.lower())
                        self.context.set_format(new_format)
                        print(f"Context format changed to: {new_format.value}")
                    except (ValueError, KeyError):
                        print("Invalid format. Use: xml, minimal, or text_only")
                    continue
                elif user_input.lower() == "/camera":
                    if self.capture_from_camera():
                        print("Image captured and ready for use in conversation")
                    else:
                        print("Failed to capture image")
                    continue
            
                # Process regular query
                response = self.process_query(user_input)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {e}")
                print(f"An error occurred: {e}")
