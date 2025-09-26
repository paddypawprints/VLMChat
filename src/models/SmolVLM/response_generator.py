"""
Response generator for SmolVLM model.

This module contains the ResponseGenerator class which handles the generation
of responses from the SmolVLM model, supporting both ONNX and standard
transformers inference with optional streaming output.
"""

import logging
from typing import List, Dict, Any
from models.SmolVLM.smol_vlm_model import SmolVLMModel

logger = logging.getLogger(__name__)

class ResponseGenerator:
    """
    Handles generation of responses from the SmolVLM model.

    This class provides a high-level interface for generating text responses
    from the SmolVLM model, supporting both ONNX runtime and standard
    transformers inference with streaming capabilities.
    """

    def __init__(self, model: SmolVLMModel):
        """
        Initialize response generator with a model instance.

        Args:
            model: Instance of SmolVLMModel to use for text generation
        """
        self._model = model

    @property
    def model(self) -> SmolVLMModel:
        """Get the underlying SmolVLM model instance."""
        return self._model
    
    def generate_response(self,
                         messages: List[Dict[str, Any]],
                         images: List[Any],
                         stream_output: bool = True) -> str:
        """
        Generate a response using the model with the provided inputs.

        Processes the input messages and images through the model to generate
        a text response. Supports both streaming and non-streaming output modes.

        Args:
            messages: List of formatted message dictionaries for conversation context
            images: List of PIL Image objects to process with the text
            stream_output: Whether to stream tokens as they're generated (affects UI responsiveness)

        Returns:
            str: Generated response text from the model

        Raises:
            Exception: Re-raises any model generation errors after logging
        """
        # Prepare model inputs from messages and images
        inputs = self._model.prepare_inputs(messages, images)

        try:
            # Choose generation method based on model configuration
            if self._model.use_onnx and stream_output:
                return self._generate_streaming_onnx(inputs)
            elif self._model.use_onnx:
                return self._generate_streaming_onnx(inputs)
            else:
                return self._model.generate_transformers(inputs)

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            raise
    
    def _generate_streaming_onnx(self, inputs: Dict[str, Any]) -> str:
        """
        Generate response with ONNX streaming support.

        Uses the ONNX runtime for efficient inference with streaming token
        generation, collecting all tokens into a complete response string.

        Args:
            inputs: Prepared model inputs dictionary

        Returns:
            str: Complete generated response text
        """
        response_tokens = []
        # Stream tokens from ONNX model and collect them
        for token_text in self._model.generate_onnx(inputs):
            response_tokens.append(token_text)
        response_tokens.append('\n')  # Add final newline
        return ''.join(response_tokens)
    
 