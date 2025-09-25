# models/model_config.py
"""
Model configuration and constants for SmolVLM.

This module defines the ModelConfig dataclass which holds all configuration
parameters needed for initializing and running the SmolVLM model, including
token limits, special tokens, and model paths.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    """
    Configuration parameters for the SmolVLM model.

    This dataclass holds all the necessary configuration parameters for model
    initialization, including paths, token limits, and special token definitions.

    Attributes:
        model_path: Path to the model (HuggingFace Hub or local path)
        max_new_tokens: Maximum number of tokens to generate in a response
        eos_token_id: Token ID representing end-of-sequence (typically newline)
        special_tokens: Dictionary mapping special token names to their string representations
    """
    model_path: str
    max_new_tokens: int = 1024
    eos_token_id: int = 198  # Default newline token for text generation
    special_tokens: Dict[str, str] = None

    def __post_init__(self):
        """
        Initialize default special tokens if none provided.

        Sets up default special tokens used by the model for conversation
        management and utterance boundaries.
        """
        if self.special_tokens is None:
            self.special_tokens = {"end_of_utterance": "<end_of_utterance>"}
