"""
Backend interface for SmolVLM backends.

Defines the abstract methods both the Transformers and ONNX backends must
implement so they present the same behavior to the SmolVLMModel facade.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Generator

from PIL import Image


class BackendBase(ABC):
    """Abstract base class for model backends."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether the backend is available for generation."""
        raise NotImplementedError()

    @abstractmethod
    def prepare_inputs(self, messages: List[Dict], images: List[Image.Image]) -> Dict[str, Any]:
        """Prepare inputs from messages and images for this backend."""

    @abstractmethod
    def generate(self, inputs: Dict[str, Any], max_new_tokens: int | None = None) -> str:
        """Generate the complete response as a string (non-streaming)."""

    @abstractmethod
    def generate_stream(self, inputs: Dict[str, Any], max_new_tokens: int | None = None) -> Generator[str, None, None]:
        """Generate tokens/strings as a stream (generator)."""
