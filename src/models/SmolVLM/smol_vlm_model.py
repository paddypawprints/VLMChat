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
from models.SmolVLM.runtime_base import RuntimeBase
from models.SmolVLM.transformers_backend import TransformersBackend
from models.SmolVLM.onnx_backend import OnnxBackend
from utils.metrics_collector import Collector, null_collector

logger = logging.getLogger(__name__)

def smol_vlm_metrics_create(collector: Collector):
    collector.register_timeseries("smolVLM-inference", ["inputs","generate","initialize","first-token"], ttl_seconds=600)
    collector.register_timeseries("smolVLM-onnx", ["vision-encoder","embeds", "generate"], ttl_seconds=600)            
    collector.register_timeseries("smolVLM-transformers", ["initialize"], ttl_seconds=600)            


class SmolVLMModel:
    """
    Facade around interchangeable backends implementing BackendBase.

    Both backends implement the same interface so behavior is consistent.
    """

    def __init__(self, config: ModelConfig, use_onnx: bool = True, collector: Optional[Collector] = null_collector()):
        self._config = config
        self._use_onnx = use_onnx
        # Optional metrics collector; models/backends may use this for telemetry
        self.collector: Optional[Collector] = collector

        # instantiate the appropriate runtime instance (only one)
        # use a plain assignment here (type comments are fine for static checkers)
        with self.collector.duration_timer("smolVLM-inference",{"initialize" : None}):
            self._runtime = self._make_runtime(self._use_onnx, config)  # type: RuntimeBase

        logger.info(f"SmolVLMModel runtime set to: {'onnx' if self._use_onnx else 'transformers'}")

    def _make_runtime(self, use_onnx: bool, config: ModelConfig) -> RuntimeBase:
        """Create and return a runtime instance based on use_onnx flag.

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

        # create transformers runtime as fallback/default
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
            # use the public collector attribute
            with self.collector.duration_timer("smolVLM-inference",{"inputs" : None}):
                # Prepare inputs and choose streaming or non-streaming generation
                inputs = self._runtime.prepare_inputs(messages, images)
            with self.collector.duration_timer("smolVLM-inference", {"generate": None}):
                if stream_output:
                    # prefer streaming generator when requested; wrap generator to record
                    # time-to-first-token using the collector's DurationTimer
                    stream_gen = self._wrap_stream_with_time_to_first_token(
                        self._runtime.generate_stream(inputs),
                        attributes={"runtime": self.current_runtime()}
                    )
                    return ''.join(token for token in stream_gen)
                else:
                    return self._runtime.generate(inputs)

        except Exception as e:
            logger.error(f"Error during generation: {e}")
            traceback.print_exc()
            raise

    def set_runtime(self, runtime: str) -> None:
        """Set the model runtime at runtime.

        runtime: one of 'onnx', 'transformers', or 'auto'. 'auto' prefers ONNX
        when available and falls back to transformers.
        """
        b = str(runtime).strip().lower()
        if b not in ('onnx', 'transformers', 'auto'):
            raise ValueError("runtime must be one of: 'onnx', 'transformers', 'auto'")

        # Recreate backend according to requested selection
        if b == 'onnx':
            # attempt to create Onnx runtime and verify availability
            runtime_candidate = self._make_runtime(True, self._config)
            if getattr(runtime_candidate, 'is_available', False):
                self._use_onnx = True
                self._runtime = runtime_candidate
            else:
                raise RuntimeError('ONNX runtime not available')
        elif b == 'transformers':
            self._use_onnx = False
            self._runtime = self._make_runtime(False, self._config)
        else:  # auto
            runtime_candidate = self._make_runtime(True, self._config)
            if getattr(runtime_candidate, 'is_available', False):
                self._use_onnx = True
                self._runtime = runtime_candidate
            else:
                self._use_onnx = False
                self._runtime = self._make_runtime(False, self._config)

    def current_runtime(self) -> str:
        """Return the currently selected runtime as a string."""
        return 'onnx' if getattr(self._runtime, 'is_available', False) and self._use_onnx else 'transformers'
    
    def _wrap_stream_with_time_to_first_token(self, gen, attributes: Optional[Dict[str, str]] = None):
        """Wrap a token generator to measure time-to-first-token.

        This starts the collector's duration timer before consuming tokens and
        stops it as soon as the first token is yielded. If no token is yielded
        the timer is stopped in the finally block.
        """
        timer = self.collector.duration_timer("smolVLM-inference", {"first-token": None})
        # Manually enter the context so we can exit on first token
        enter_result = None
        try:
            enter_result = timer.__enter__()
        except Exception:
            # Ensure sampling doesn't fail generation
            pass

        first = True
        try:
            for token in gen:
                if first:
                    try:
                        timer.__exit__(None, None, None)
                    except Exception:
                        pass
                    first = False
                yield token
        finally:
            # Ensure timer is closed if no tokens were produced or generator ended
            if first:
                try:
                    timer.__exit__(None, None, None)
                except Exception:
                    pass

        
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
    
