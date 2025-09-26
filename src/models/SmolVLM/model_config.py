# models/model_config.py
"""
Model configuration and constants for SmolVLM.

This module defines the ModelConfig dataclass which holds all configuration
parameters needed for initializing and running the SmolVLM model, including
token limits, special tokens, and model paths. Configuration values are now
loaded from the global application configuration.
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class ModelConfig:
    """
    Configuration parameters for the SmolVLM model.

    This dataclass holds all the necessary configuration parameters for model
    initialization, including paths, token limits, and special token definitions.
    Values are loaded from the global application configuration.

    Attributes:
        model_path: Path to the model (HuggingFace Hub or local path)
        max_new_tokens: Maximum number of tokens to generate in a response
        eos_token_id: Token ID representing end-of-sequence (typically newline)
        special_tokens: Dictionary mapping special token names to their string representations
    """
    model_path: str
    max_new_tokens: int = None
    eos_token_id: int = None
    special_tokens: Dict[str, str] = None

    def __post_init__(self):
        """
        Initialize configuration from global config and set up default special tokens.

        Loads configuration values from the global application configuration
        and sets up default special tokens used by the model for conversation
        management and utterance boundaries.
        """
        # Import here to avoid circular imports
        from config import get_config

        config = get_config()

        # Load values from global config if not explicitly provided
        if self.max_new_tokens is None:
            self.max_new_tokens = config.model.max_new_tokens
        if self.eos_token_id is None:
            self.eos_token_id = config.model.eos_token_id

        # Initialize default special tokens if none provided
        if self.special_tokens is None:
            self.special_tokens = {"end_of_utterance": "<end_of_utterance>"}
