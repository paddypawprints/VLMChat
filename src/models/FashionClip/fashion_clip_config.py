"""
FashionClip model configuration.

This module provides configuration management for the FashionClip model,
including model selection, device settings, and runtime options.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Any
from utils.config import VLMChatConfig

logger = logging.getLogger(__name__)


@dataclass
class FashionClipConfig:
    """Configuration for FashionClip model."""
    model_name: str
    pretrained_model_path: str = ""
    device: str = "cpu"
    model_kwargs: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.model_kwargs is None:
            self.model_kwargs = {}


def get_fashion_clip_config(config: VLMChatConfig) -> FashionClipConfig:
    """
    Create a FashionClipConfig from the application config.
    
    Args:
        config: The main application configuration
        
    Returns:
        FashionClipConfig with settings from the application config
    """
    # Default to Marqo's FashionSigLIP model
    model_name = getattr(config.model, 'fashion_clip_model_name', 'hf-hub:Marqo/marqo-fashionSigLIP')
    pretrained = getattr(config.model, 'fashion_clip_pretrained', '')
    device = getattr(config.model, 'device', 'cpu')
    
    logger.info(f"FashionClip config: model={model_name}, device={device}")
    
    return FashionClipConfig(
        model_name=model_name,
        pretrained_model_path=pretrained,
        device=device,
        model_kwargs={}
    )
