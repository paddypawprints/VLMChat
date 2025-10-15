# models/smol_vlm_model.py
"""
SmolVLM model wrapper with ONNX runtime support.

This module provides the SmolVLMModel class which wraps the HuggingFace
SmolVLM model with optional ONNX runtime support for improved inference
performance. It handles model loading, input preparation, and text generation
with support for both standard transformers and ONNX execution.
"""

import logging
import traceback
from typing import List, Dict, Any
from PIL import Image
from prompt.prompt import Prompt

from models.SmolVLM.model_config import ModelConfig
from models.SmolVLM.backend_base import BackendBase
from models.SmolVLM.transformers_backend import TransformersBackend
from models.SmolVLM.onnx_backend import OnnxBackend

logger = logging.getLogger(__name__)


class SmolVLMModel:
    """
    Facade around interchangeable backends implementing BackendBase.

    Both backends implement the same interface so behavior is consistent.
    """

    def __init__(self, config: ModelConfig, use_onnx: bool = True):
        self._config = config
        self._use_onnx = use_onnx

        # instantiate backends
        self._transformers: BackendBase = TransformersBackend(config.model_path, config)
        self._onnx: BackendBase = OnnxBackend(config)
        if not self._onnx.is_available:
            self._use_onnx = False

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def use_onnx(self) -> bool:
        return self._use_onnx

    def prepare_onnx_inputs(self, messages: List[Dict], images: List[Image.Image]):
        return self._onnx.prepare_inputs(messages, images)

    def prepare_transformers_inputs(self, messages: List[Dict], images: List[Image.Image]):
        return self._transformers.prepare_inputs(messages, images)

    def generate_onnx(self, inputs: Dict[str, Any], max_new_tokens: int = None):
        return self._onnx.generate_stream(inputs, max_new_tokens=max_new_tokens)

    def generate_transformers(self, inputs: Dict[str, Any], max_new_tokens: int = None) -> str:
        return self._transformers.generate(inputs, max_new_tokens=max_new_tokens)

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

        try:
            # Choose generation method based on model configuration
            if self._use_onnx and stream_output:
                inputs = self.prepare_onnx_inputs(messages, images)
                return self._generate_streaming_onnx(inputs)
            elif self._use_onnx:
                inputs = self.prepare_onnx_inputs(messages, images)
                return self._generate_streaming_onnx(inputs)
            else:
                inputs = self.prepare_transformers_inputs(messages, images)
                return self.generate_transformers(inputs)

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            traceback.print_exc()
            raise

    def set_backend(self, backend: str) -> None:
        """Set the model backend at runtime.

        backend: one of 'onnx', 'transformers', or 'auto'. 'auto' prefers ONNX
        when available and falls back to transformers.
        """
        b = str(backend).strip().lower()
        if b not in ('onnx', 'transformers', 'auto'):
            raise ValueError("backend must be one of: 'onnx', 'transformers', 'auto'")

        if b == 'onnx':
            if getattr(self._onnx, 'is_available', False):
                self._use_onnx = True
            else:
                raise RuntimeError('ONNX backend not available')
        elif b == 'transformers':
            self._use_onnx = False
        else:  # auto
            self._use_onnx = getattr(self._onnx, 'is_available', False)

        logger.info(f"SmolVLMModel backend set to: {'onnx' if self._use_onnx else 'transformers'}")

    def current_backend(self) -> str:
        """Return the currently selected backend as a string."""
        return 'onnx' if self._use_onnx else 'transformers'
    
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
        for token_text in self.generate_onnx(inputs):
            response_tokens.append(token_text)
        response_tokens.append('\n')  # Add final newline
        return ''.join(response_tokens)
    
    def get_messages(self, prompt: Prompt) -> List[Dict[str, Any]]:
        """
        Convert a Prompt object into the message format expected by the model.

        Formats the conversation history and current user input into the structured
        message format required by the chat template processor.

        Args:
            prompt: Prompt object containing conversation history and current input

        Returns:
            List[Dict[str, Any]]: List of formatted message dictionaries
        """
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "You are a helpful assistant."},
                    {"type": "text", "text": prompt.history.get_formatted_history()},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt.user_input},
                    {"type": "image", "image": ""},
                ]
            },
        ]
        return messages
    
