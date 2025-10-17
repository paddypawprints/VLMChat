# main/chat_application.py
"""
Main chat application orchestrating all components.

This module contains the SmolVLMChatApplication class which serves as the main
coordinator for the chat application. It integrates the SmolVLM model, prompt
handling, image processing, camera capture, and user interface components.
"""

import logging
from PIL import Image

from models.SmolVLM.smol_vlm_model import SmolVLMModel, smol_vlm_metrics_create
from models.SmolVLM.model_config import ModelConfig
from utils.image_utils import load_image_from_url, load_image_from_file
from src.prompt.prompt import Prompt
from src.prompt.history_format import HistoryFormat
#from utils.camera import IMX500ObjectDetection
from utils.camera_factory import CameraFactory
from utils.camera_base import BaseCamera, CameraModel, Platform
from utils.metrics_collector import Session, CounterInstrument, HistogramByAttributeInstrument  

logger = logging.getLogger(__name__)

from main.service_response import ServiceResponse
from main.service_response import ServiceResponse as SR
# ServiceResponse codes are defined in `src/main/service_response.py`.
# For quick reference:
#   SR.Code.OK (0)                - Success
#   SR.Code.EXIT (1)              - Exit interactive loop
#   SR.Code.IMAGE_LOAD_FAILED (2) - Image URL/file load failed
#   SR.Code.INVALID_FORMAT (3)    - Invalid /format argument
#   SR.Code.CAMERA_FAILED (4)     - Camera capture failed
#   SR.Code.NO_METRICS_SESSION (5)- No metrics session available
#   SR.Code.BACKEND_FAILED (6)    - Backend query or switch failed
#   SR.Code.UNKNOWN_COMMAND (7)   - Unrecognized command

class SmolVLMChatApplication:
    """Main application class for SmolVLM chat interface."""
    
    def __init__(self,
                 model_path: str = None,
                 use_onnx: bool = None,
                 max_pairs: int = None,
                 max_images: int = None,
                 history_format: HistoryFormat = HistoryFormat.XML):
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
            self._collector.register_timeseries("camera", ["inputs","generate"], ttl_seconds=600)
            smol_vlm_metrics_create(self._collector)
            self._session = self._collector and self._collector and None
            try:
                from utils.metrics_collector import Session
                self._session = Session(self._collector)
            except Exception:
                logger.exception("Failed to create metrics sessions")

            counter = CounterInstrument("requests_counter", ["generate"])
            self._session.add_instrument(counter, "smolVLM-inference")
            histogram = HistogramByAttributeInstrument("his")
            self._session.add_instrument(histogram, "smolVLM-inference")
            histogram_onnx = HistogramByAttributeInstrument("his-onnx")
            self._session.add_instrument(histogram_onnx, "smolVLM-onnx")

            self._model = SmolVLMModel(self._config, use_onnx=use_onnx, collector=self._collector)

            # Create an application-level session and a model-level session
 
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

    # --- Service methods (business logic) ---------------------------------
    # Note: help UI is handled by console_io (app._print_help_message is used)

    def _service_load_url(self, url: str) -> ServiceResponse:
        image = load_image_from_url(url)
        if image:
            self._prompt.history.set_current_image(image)
            self._prompt.history.clear_history()
            return ServiceResponse(ServiceResponse.Code.OK, "Image loaded successfully. Conversation history cleared for new image.")
        return ServiceResponse(ServiceResponse.Code.IMAGE_LOAD_FAILED, "Failed to load image.")

    def _service_load_file(self, path: str) -> ServiceResponse:
        image = load_image_from_file(path)
        if image:
            self._prompt.history.set_current_image(image)
            self._prompt.history.clear_history()
            return ServiceResponse(ServiceResponse.Code.OK, "Image loaded successfully. Conversation history cleared for new image.")
        return ServiceResponse(ServiceResponse.Code.IMAGE_LOAD_FAILED, "Failed to load image.")

    def _service_clear_context(self) -> ServiceResponse:
        self._prompt.history.clear_history()
        return ServiceResponse(ServiceResponse.Code.OK, "Conversation history cleared.")

    def _service_show_context(self) -> ServiceResponse:
        return ServiceResponse(ServiceResponse.Code.OK, str(self._prompt.history))

    def _service_context_stats(self) -> ServiceResponse:
        stats = self._prompt.history.get_stats()
        lines = ["Context Buffer Statistics:"]
        for key, value in stats.items():
            lines.append(f"  {key}: {value}")
        return ServiceResponse(ServiceResponse.Code.OK, "\n".join(lines))

    def _service_format(self, arg: str) -> ServiceResponse:
        try:
            new_format = HistoryFormat(arg.strip().lower())
            self._prompt.history.set_format(new_format)
            return ServiceResponse(ServiceResponse.Code.OK, f"Context format changed to: {new_format.value}")
        except (ValueError, KeyError):
            return ServiceResponse(ServiceResponse.Code.INVALID_FORMAT, "Invalid format. Use: xml or minimal")

    def _service_camera(self) -> ServiceResponse:
        if self.capture_from_camera():
            return ServiceResponse(ServiceResponse.Code.OK, "Image captured and ready for use in conversation")
        return ServiceResponse(ServiceResponse.Code.CAMERA_FAILED, "Failed to capture image")

    def _service_metrics(self) -> ServiceResponse:
        sess = getattr(self, '_session', None)
        if sess is None:
            return ServiceResponse(ServiceResponse.Code.NO_METRICS_SESSION, "No metrics session")
        sess_dict = sess.to_dict()
        out_lines = ["=== Session Metrics ==="]
        out_lines.append(f"start_time: {sess_dict.get('start_time')}")
        out_lines.append(f"end_time: {sess_dict.get('end_time')}")
        insts = sess_dict.get('instruments', [])
        if not insts:
            out_lines.append("No instruments attached to the session.")
        else:
            for item in insts:
                ts_name = item.get('timeseries')
                inst_export = item.get('instrument')
                out_lines.append(f"Instrument bound to timeseries '{ts_name}':")
                try:
                    import json as _json
                    out_lines.append(_json.dumps(inst_export, indent=2))
                except Exception:
                    out_lines.append(str(inst_export))
        return ServiceResponse(ServiceResponse.Code.OK, "\n".join(out_lines))

    def _service_backend(self, parts: list[str]) -> ServiceResponse:
        if len(parts) == 1:
            try:
                return ServiceResponse(ServiceResponse.Code.OK, f"Current backend: {self._model.current_backend()}")
            except Exception:
                return ServiceResponse(ServiceResponse.Code.BACKEND_FAILED, "Failed to query current backend")
        new_backend = parts[1].strip().lower()
        try:
            self._model.set_runtime(new_backend)
            return ServiceResponse(ServiceResponse.Code.OK, f"Backend switched to: {self._model.current_backend()}")
        except Exception as e:
            return ServiceResponse(ServiceResponse.Code.BACKEND_FAILED, f"Failed to switch backend: {e}")

    # _process_command was removed and the interactive loop moved to
    # main.console_io.run_interactive_chat(app). This class retains only the
    # service methods; callers should use the console_io helpers to run the
    # interactive loop.

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
            logger.info(f"Captured image saved to: {filepath}")
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
        logger.info("Available Commands:")
        logger.info("  /load_url <url>     - Load image from URL for conversation")
        logger.info("  /load_file <path>   - Load image from local file path")
        logger.info("  /clear_context      - Clear conversation history")
        logger.info("  /show_context       - Display current conversation history")
        logger.info("  /context_stats      - Show context buffer statistics")
        logger.info("  /format <format>    - Change history format (xml|minimal|nohistory)")
        logger.info("  /camera             - Capture image from camera")
        logger.info("  /help               - Show this help message")
        logger.info("  /quit               - Exit the application")

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
        # NOTE: The interactive loop is owned by the console_io module so this
        # class intentionally does not import or depend on console_io. To run
        # the interactive interface, call `main.console_io.run_interactive_chat()`
        # which will instantiate and use this application class.
        raise RuntimeError("Interactive loop is owned by console_io; call run_interactive_chat() from main.console_io")
