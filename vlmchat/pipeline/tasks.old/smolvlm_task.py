"""
SmolVLM task adapter for pipeline integration.

Creates and manages a SmolVLM model instance, runs inference on image + prompt.
"""

from typing import Optional, Dict, Any
from PIL import Image
import logging

from ..task_base import BaseTask, Context, ContextDataType, register_task
from ...models.SmolVLM.smol_vlm_model import SmolVLMModel
from ...utils.config import VLMChatConfig

logger = logging.getLogger(__name__)


@register_task('smolvlm')
class SmolVLMTask(BaseTask):
    """
    Pipeline task adapter for SmolVLM inference.
    
    Reads IMAGE and TEXT from context (expects formatted_history + user_input),
    runs model inference, and appends response to TEXT.
    Creates its own SmolVLM model instance on first use.
    """
    
    def __init__(self, task_id: str = "smolvlm", config: Optional[VLMChatConfig] = None, image_format: str = "pil"):
        """
        Initialize SmolVLM task.
        
        Args:
            task_id: Unique identifier for this task
            config: Optional VLMChatConfig, will load default if not provided
            image_format: Expected image format ("pil" is currently the only supported format)
        """
        super().__init__(task_id)
        self._config = config
        self._model = None  # Lazy initialization
        self.system_prompt = "You are a helpful assistant."
        self.runtime = None  # Will be set by configure if specified
        self.image_format = image_format.lower()
        
        # SmolVLM only supports PIL images currently
        if self.image_format != "pil":
            raise ValueError(f"Unsupported image format: {self.image_format}. SmolVLM only supports 'pil'.")
        
        # Define contracts
        # Input: Requires IMAGE and TEXT (history + user input)
        # Output: Produces TEXT (response)
        self.input_contract = {
            ContextDataType.IMAGE: Image.Image,
            ContextDataType.TEXT: str
        }
        self.output_contract = {ContextDataType.TEXT: str}
        
        # Set native format from backend (will be updated when model is initialized)
        from ..image_format import ImageFormat
        self.native_input_format = ImageFormat.PIL
    
    @property
    def model(self) -> SmolVLMModel:
        """Lazy load the model on first access."""
        if self._model is None:
            logger.info(f"Task {self.task_id}: Initializing SmolVLM model...")
            if self._config is None:
                raise RuntimeError(f"Task {self.task_id}: No config available. Config must be set via context before using SmolVLM.")
            self._model = SmolVLMModel(self._config, collector=self.collector)
            
            # Update native_input_format from backend after model is initialized
            if hasattr(self._model, '_runtime') and hasattr(self._model._runtime, 'native_image_format'):
                from ..image_format import ImageFormat
                backend_format = self._model._runtime.native_image_format
                if backend_format:
                    format_map = {
                        "pil": ImageFormat.PIL,
                        "numpy": ImageFormat.NUMPY,
                        "torch_cpu": ImageFormat.TORCH_CPU,
                        "torch_gpu": ImageFormat.TORCH_GPU
                    }
                    self.native_input_format = format_map.get(backend_format, ImageFormat.PIL)
            
            logger.info(f"Task {self.task_id}: SmolVLM model initialized")
        return self._model
    
    def configure(self, **params) -> None:
        """
        Configure SmolVLM task from DSL parameters.
        
        Args:
            **params: Keyword arguments with configuration
                - system_prompt: System prompt for model behavior (default: "You are a helpful assistant.")
                - runtime: Model runtime backend ("onnx", "transformers", or "auto")
        
        Example:
            smolvlm(system_prompt="You are a vision expert.", runtime="onnx")
        """
        if "system_prompt" in params:
            self.system_prompt = params["system_prompt"]
        
        if "runtime" in params:
            self.runtime = params["runtime"]
            # Set the runtime on the model
            self.model.set_runtime(self.runtime)
    
    def run(self, context: Context) -> Context:
        """
        Run SmolVLM inference on image and text from context.
        
        Expects TEXT list to contain at least 2 items:
        - TEXT[-2]: formatted conversation history (may be empty)
        - TEXT[-1]: current user input
        
        Args:
            context: Pipeline context containing IMAGE and TEXT
            
        Returns:
            Updated context with TEXT (response) appended
        """
        # Get config and collector from context if not already set
        if self._config is None and context.config:
            self._config = context.config
        if self.collector is None and context.collector:
            self.collector = context.collector
        
        # Get inputs from context
        if ContextDataType.IMAGE not in context.data or not context.data[ContextDataType.IMAGE]:
            raise ValueError(f"Task {self.task_id}: IMAGE not found in context")
        if ContextDataType.TEXT not in context.data or not context.data[ContextDataType.TEXT]:
            raise ValueError(f"Task {self.task_id}: TEXT not found in context")
        
        text_list = context.data[ContextDataType.TEXT]
        logger.debug(f"Task {self.task_id}: Received text_list with {len(text_list)} items: {text_list}")
        if len(text_list) < 2:
            raise ValueError(f"Task {self.task_id}: Expected at least 2 TEXT items (history + user_input), got {len(text_list)}")
        
        # Get most recent image (may be ImageContainer or raw image)
        from ..image_container import ImageContainer
        from ..image_format import ImageFormat
        image_item = context.data[ContextDataType.IMAGE][-1]
        
        if isinstance(image_item, ImageContainer) and self.native_input_format:
            # Extract PIL image from container (runner already converted to our native format)
            image = image_item.get_format(self.native_input_format)
            if image is None:
                # Fallback to any available format
                for fmt in image_item.get_cached_formats():
                    image = image_item.get_format(fmt)
                    if image is not None:
                        break
        else:
            # Fallback for raw images
            image = image_item
        formatted_history = text_list[-2]  # Conversation history from history task
        user_input = text_list[-1]  # Current user input
        
        # Build messages for SmolVLM
        # Skip history in system message if it's empty
        system_content = [{"type": "text", "text": self.system_prompt}]
        if formatted_history:  # Only add history if non-empty
            system_content.append({"type": "text", "text": formatted_history})
        
        messages = [
            {
                "role": "system",
                "content": system_content
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_input},
                    {"type": "image", "image": ""},
                ]
            },
        ]
        
        # Generate response
        response = self.model.generate_response(
            messages=messages,
            images=[image],
            stream_output=False  # Non-streaming for pipeline
        )
        
        # Append response to TEXT context
        context.data[ContextDataType.TEXT].append(response)
        
        return context
    
    def describe(self) -> str:
        """Return description of what this task does."""
        return "Runs SmolVLM inference on IMAGE and TEXT (history + user_input), appends response to TEXT"
    
    def describe_parameters(self) -> Dict[str, str | Dict[str, Any]]:
        """Return parameter descriptions for SmolVLM configuration."""
        return {
            "system_prompt": {
                "description": "System prompt defining model behavior",
                "type": "str",
                "default": "You are a helpful assistant.",
                "example": "You are a vision expert that describes images in detail."
            },
            "runtime": {
                "description": "Model runtime backend",
                "type": "str",
                "choices": ["onnx", "transformers", "auto"],
                "default": "auto",
                "example": "onnx"
            }
        }
