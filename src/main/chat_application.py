# main/chat_application.py
"""
Main chat application orchestrating all components.

This module contains the SmolVLMChatApplication class which serves as the main
coordinator for the chat application. It integrates the SmolVLM model, prompt
handling, image processing, camera capture, and user interface components.
"""

import logging
from PIL import Image

from models.SmolVLM.smol_vlm_model import SmolVLMModel
from models.SmolVLM.model_config import ModelConfig
from utils.image_utils import load_image_from_url, load_image_from_file
from prompt.prompt import Prompt,HistoryFormat
#from utils.camera import IMX500ObjectDetection
from utils.camera_factory import CameraFactory
from utils.camera_base import BaseCamera, CameraModel, Platform

logger = logging.getLogger(__name__)

class SmolVLMChatApplication:
    """Main application class for SmolVLM chat interface."""
    
    def __init__(self,
                 model_path: str = None,
                 use_onnx: bool = None,
                 max_pairs: int = None,
                 max_images: int = None,
                 history_format: HistoryFormat = None):
        """
        Initialize the chat application with all required components.

        Sets up the model configuration, loads the SmolVLM model with optional ONNX
        runtime support, initializes the response generator, conversation history
        manager, and camera interface. Configuration values are loaded from the
        global application configuration if not explicitly provided.

        Args:
            model_path: Path to the SmolVLM model on HuggingFace Hub or local path (optional)
            use_onnx: Whether to use ONNX runtime for faster inference (optional)
            max_pairs: Maximum number of conversation pairs to keep in history (optional)
            max_images: Maximum number of images to keep in context (optional)
            history_format: Format for conversation history (XML, MINIMAL) (optional)

        Raises:
            Exception: If model loading or component initialization fails
        """
        # Import here to avoid circular imports
        from config import get_config

        # Get global configuration
        config = get_config()

        # Configure logging from global configuration
        logging.basicConfig(
            level=getattr(logging, config.logging.level),
            format=config.logging.format
        )

        # Use configuration values if parameters not provided
        if model_path is None:
            model_path = config.model.model_path
        if use_onnx is None:
            use_onnx = config.model.use_onnx
        if max_pairs is None:
            max_pairs = config.conversation.max_pairs
        if max_images is None:
            max_images = config.conversation.max_images
            if history_format is None:
                # Convert string to enum
                from prompt.history_format import HistoryFormat as ConfigHistoryFormat
                history_format = ConfigHistoryFormat(config.conversation.history_format.value)

            # Initialize core model components
            self._config = ModelConfig(model_path=model_path)
            # Create a metrics collector and pass it into the model for telemetry
            from utils.metrics_collector import Collector
            self._collector = Collector()
            self._model = SmolVLMModel(self._config, use_onnx=use_onnx, collector=self._collector)

            # Create an application-level session and a model-level session
            self._app_session = self._collector and self._collector and None
            try:
                from utils.metrics_collector import Session
                self._app_session = Session(self._collector)
                # create a separate session for model-internal metrics
                self._model.session = Session(self._collector)
            except Exception:
                logger.exception("Failed to create metrics sessions")

            # Initialize conversation management
            self._prompt = Prompt(
                max_pairs=max_pairs,
                max_images=max_images,
                history_format=history_format
            )

        # Initialize hardware interfaces
        # Resolve platform from loaded configuration (prefer explicit runtime_platform)
        detected_platform = None
        try:
            detected_platform = config.get_runtime_platform()
        except Exception:
            detected_platform = None

        # If detection returned None, fall back to Platform.RPI
        resolved_platform = detected_platform or Platform.RPI

        logger.info(f"Using platform for camera creation: {resolved_platform}")

        # Create camera for the resolved platform. Do not pass a model so the
        # CameraFactory will select the platform-appropriate default unless an
        # explicit override is provided elsewhere.
        default_map = CameraFactory.get_default_camera_for_platform(resolved_platform)
        logger.info(f"CameraFactory default for {resolved_platform}: {default_map}")

        self._camera = CameraFactory.create_camera(
            platform=resolved_platform,
            # with_detection left unspecified so factory uses its default
        )
        
        logger.info("SmolVLM Chat Application initialized successfully")

    @property
    def config(self) -> ModelConfig:
        """Get the model configuration."""
        return self._config

    @property
    def model(self) -> SmolVLMModel:
        """Get the SmolVLM model instance."""
        return self._model

    @property
    def prompt(self) -> Prompt:
        """Get the prompt manager."""
        return self._prompt

    @property
    def camera(self) -> BaseCamera:
        """Get the camera interface."""
        return self._camera

    def set_context_format(self, history_format: HistoryFormat) -> None:
        """
        Set the conversation history format.

        Args:
            history_format: New format to use for conversation history
        """
        self._prompt.history.set_format(history_format)

    def capture_from_camera(self) -> bool:
        """
        Capture an image from the camera and load it into the current context.

        Uses the IMX500 camera interface to capture a single image, saves it to disk,
        and loads it into the conversation context for use in subsequent queries.

        Returns:
            bool: True if capture successful, False otherwise

        Raises:
            Exception: Camera or image processing errors are caught and logged
        """
        try:
            filepath, image = self._camera.capture_single_image()
            self._prompt.current_image = image
            print(f"Captured image saved to: {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to capture image: {e}")
            return False


    def process_query(self, user_input: str, stream_output: bool = True) -> str:
        """
        Process a user query with the current image and conversation context.

        Takes a user's text input, combines it with the current image and conversation
        history to generate a contextually aware response using the SmolVLM model.

        Args:
            user_input: The user's text query or question
            stream_output: Whether to stream the response as it's generated

        Returns:
            str: The model's generated response text

        Raises:
            Exception: Model generation errors are caught and returned as error messages
        """
        # Validate that an image is loaded before processing
        if not self._prompt.current_image:
            return "No image loaded. Please load an image first."

        logger.info("Processing user query with context")

        # Update the prompt with current user input
        self._prompt._user_input = user_input
        messages = self._model.get_messages(self._prompt)

        # Generate response using the model
        try:
            response = self._model.generate_response(
                messages=messages,
                images=[self._prompt.current_image],
                stream_output=stream_output
            )

            # Add this interaction to conversation history
            self._prompt.history.add_conversation_pair(
                request_text=user_input,
                response_text=response
            )

            logger.info(f"Context stats: {self._prompt.history.get_stats()}")
            return response

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return f"Error generating response: {e}"
    
    def _print_help_message(self) -> None:
        """
        Display available commands and their descriptions.

        Prints a formatted list of all available slash commands that users can
        use to interact with the chat application, including image loading,
        context management, and application control commands.
        """
        print("Available Commands:")
        print("  /load_url <url>     - Load image from URL for conversation")
        print("  /load_file <path>   - Load image from local file path")
        print("  /clear_context      - Clear conversation history")
        print("  /show_context       - Display current conversation history")
        print("  /context_stats      - Show context buffer statistics")
        print("  /format <format>    - Change history format (xml|minimal)")
        print("  /camera             - Capture image from camera")
        print("  /help               - Show this help message")
        print("  /quit               - Exit the application")

    def run_interactive_chat(self) -> None:
        """
        Run the interactive chat loop.

        Starts the main user interface loop that handles user input, processes
        commands and queries, and manages the conversation flow. Continues until
        the user exits with /quit or interrupts with Ctrl+C.

        Raises:
            KeyboardInterrupt: Handled gracefully to allow clean exit
            Exception: Other exceptions are logged but don't crash the application
        """
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
                    # Extract URL from command
                    url = user_input[10:].strip()
                    image = load_image_from_url(url)
                    if image:
                        print("Image loaded successfully!")
                        self._prompt.history.set_current_image(image)
                        self._prompt.history.clear_history()
                        print("Conversation history cleared for new image.")
                    else:
                        print("Failed to load image.")
                    continue
                elif user_input.startswith('/load_file '):
                    # Extract file path from command
                    path = user_input[11:].strip()
                    image = load_image_from_file(path)
                    if image:
                        print("Image loaded successfully!")
                        self._prompt.history.set_current_image(image)
                        self._prompt.history.clear_history()
                        print("Conversation history cleared for new image.")
                    else:
                        print("Failed to load image.")
                    continue
                elif user_input.startswith('/clear_context'):
                    # Clear all conversation history
                    self._prompt.history.clear_history()
                    print("Conversation history cleared.")
                    continue
                elif user_input.startswith('/show_context'):
                    # Display the current conversation history
                    print(str(self._prompt.history))
                    continue
                elif user_input.startswith('/context_stats'):
                    # Show detailed statistics about conversation context
                    stats = self._prompt.history.get_stats()
                    print("Context Buffer Statistics:")
                    for key, value in stats.items():
                        print(f"  {key}: {value}")
                    continue
                elif user_input.startswith("/format"):
                    # Change the conversation history formatting
                    try:
                        _, format_name = user_input.split(maxsplit=1)
                        new_format = HistoryFormat(format_name.lower())
                        self._prompt.history.set_format(new_format)
                        print(f"Context format changed to: {new_format.value}")
                    except (ValueError, KeyError):
                        print("Invalid format. Use: xml or minimal")
                    continue
                elif user_input.lower() == "/camera":
                    # Capture image from camera
                    if self.capture_from_camera():
                        print("Image captured and ready for use in conversation")
                    else:
                        print("Failed to capture image")
                    continue
                elif user_input.startswith('/backend'):
                    # Show or change the model backend
                    parts = user_input.split()
                    if len(parts) == 1:
                        print(f"Current backend: {self._model.current_backend()}")
                    else:
                        new_backend = parts[1].strip().lower()
                        try:
                            self._model.set_backend(new_backend)
                            print(f"Backend switched to: {self._model.current_backend()}")
                        except Exception as e:
                            print(f"Failed to switch backend: {e}")
                    continue

                # Process regular query (not a command)
                response = self.process_query(user_input)
                print(f"\nSmolVLM: {response}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            #except Exception as e:
                #logger.error(f"Error in chat loop: {e}")
                #print(f"An error occurred: {e}")
