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
from typing import List, Dict, Any, Optional
from PIL import Image
from prompt.prompt import Prompt

from models.SmolVLM.model_config import ModelConfig
from models.SmolVLM.backend_base import BackendBase
from models.SmolVLM.transformers_backend import TransformersBackend
from models.SmolVLM.onnx_backend import OnnxBackend
from utils.metrics_collector import Collector

logger = logging.getLogger(__name__)


class SmolVLMModel:
    """
    Facade around interchangeable backends implementing BackendBase.

    Both backends implement the same interface so behavior is consistent.
    """

    def __init__(self, config: ModelConfig, use_onnx: bool = True, collector: Optional[Collector] = None):
        self._config = config
        self._use_onnx = use_onnx
        # Optional metrics collector; models/backends may use this for telemetry
        self.collector: Optional[Collector] = collector

        # instantiate the appropriate backend instance (only one)
        self._backend: BackendBase = self._make_backend(self._use_onnx, config)

    def _make_backend(self, use_onnx: bool, config: ModelConfig) -> BackendBase:
        """Create and return a backend instance based on use_onnx flag.

        If ONNX creation fails or reports not available, fall back to Transformers.
        """
        if use_onnx:
            try:
                ob = OnnxBackend(config)
                if getattr(ob, "is_available", False):
                    return ob
                # fallthrough to transformers
            except Exception:
                logger.exception("Failed to create OnnxBackend; falling back to Transformers")

        # create transformers backend as fallback/default
        try:
            tb = TransformersBackend(config.model_path, config)
            return tb
        except Exception:
            logger.exception("Failed to create TransformersBackend")
            # Re-raise to surface initialization error
            raise

    @property
    def config(self) -> ModelConfig:
        return self._config

    @property
    def use_onnx(self) -> bool:
        return self._use_onnx

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
            # Prepare inputs and choose streaming or non-streaming generation
            inputs = self._backend.prepare_inputs(messages, images)
            if stream_output:
                # prefer streaming generator when requested
                return ''.join([token for token in self._backend.generate_stream(inputs)])
            else:
                return self._backend.generate(inputs)

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

        # Recreate backend according to requested selection
        if b == 'onnx':
            # attempt to create OnnxBackend and verify availability
            backend_candidate = self._make_backend(True, self._config)
            if getattr(backend_candidate, 'is_available', False):
                self._use_onnx = True
                self._backend = backend_candidate
            else:
                raise RuntimeError('ONNX backend not available')
        elif b == 'transformers':
            self._use_onnx = False
            self._backend = self._make_backend(False, self._config)
        else:  # auto
            backend_candidate = self._make_backend(True, self._config)
            if getattr(backend_candidate, 'is_available', False):
                self._use_onnx = True
                self._backend = backend_candidate
            else:
                self._use_onnx = False
                self._backend = self._make_backend(False, self._config)

        logger.info(f"SmolVLMModel backend set to: {'onnx' if self._use_onnx else 'transformers'}")

    def current_backend(self) -> str:
        """Return the currently selected backend as a string."""
        return 'onnx' if getattr(self._backend, 'is_available', False) and self._use_onnx else 'transformers'
    
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
        # Legacy helper retained for API compatibility: delegate to backend
        response_tokens = []
        for token_text in self._backend.generate_stream(inputs):
            response_tokens.append(token_text)
        response_tokens.append('\n')
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
    
