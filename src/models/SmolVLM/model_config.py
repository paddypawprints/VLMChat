# models/model_config.py
"""Model configuration and constants."""

from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class ModelConfig:
    """Configuration for the SmolVLM model."""
    model_path: str
    max_new_tokens: int = 1024
    eos_token_id: int = 198  # Default newline token
    special_tokens: Dict[str, str] = None
    
    def __post_init__(self):
        if self.special_tokens is None:
            self.special_tokens = {"end_of_utterance": "<end_of_utterance>"}
