"""
Runtime base interface for SmolVLM backends.

Defines the abstract methods that SmolVLM backends (Transformers, ONNX) must implement.
Extends BaseRuntime from model_base with SmolVLM-specific methods.
"""
from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Generator

from PIL import Image
from models.model_base import BaseRuntime


class SmolVLMRuntimeBase(BaseRuntime):
    """Abstract base class for SmolVLM runtime backends."""

    @abstractmethod
    def prepare_inputs(self, messages: List[Dict], images: List[Image.Image]) -> Dict[str, Any]:
        """Prepare inputs from messages and images for this backend."""
        raise NotImplementedError()

    @abstractmethod
    def generate(self, inputs: Dict[str, Any], max_new_tokens: int | None = None) -> str:
        """Generate the complete response as a string (non-streaming)."""
        raise NotImplementedError()

    @abstractmethod
    def generate_stream(self, inputs: Dict[str, Any], max_new_tokens: int | None = None) -> Generator[str, None, None]:
        """Generate tokens/strings as a stream (generator)."""
        raise NotImplementedError()
