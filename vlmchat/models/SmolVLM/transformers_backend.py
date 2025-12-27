"""
Transformers backend implementing the BackendBase interface.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Generator, Optional

from utils.config import VLMChatConfig
import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

from PIL import Image

from models.SmolVLM.runtime_base import SmolVLMRuntimeBase
from models.SmolVLM.model_config import ModelConfig

from metrics.metrics_collector import Collector, null_collector

logger = logging.getLogger(__name__)


class TransformersBackend(SmolVLMRuntimeBase):
    def __init__(self, config: VLMChatConfig,  collector: Collector = null_collector()):
        super().__init__(config)
        self._config = ModelConfig(config)
        self._model_path = config.model.model_path
        self._available = True
        self._collector = collector
        collector.register_timeseries("smolVLM-transformers", ["initialize"], ttl_seconds=600)            
        try:
            with self._collector.duration_timer("SmolVLM-transformers", {"initialize" : ""}):
                self._model_config = AutoConfig.from_pretrained(self._model_path)
                self._processor = AutoProcessor.from_pretrained(self._model_path)
                self._model = AutoModelForImageTextToText.from_pretrained(
                    self._model_path,
                    torch_dtype=torch.bfloat16,
                )
                self._image_processor = self._processor.image_processor
                self._tokenizer = self._processor.tokenizer
        except Exception as e:
            logger.warning(f"Failed to initialize TransformersBackend: {e}")
            self._available = False

    @property
    def native_image_format(self) -> str:
        """SmolVLM Transformers backend expects PIL images."""
        return "pil"

    @property
    def is_available(self) -> bool:
        return self._available

    def prepare_inputs(self, messages: List[Dict], images: List[Image.Image]) -> Dict[str, Any]:
        if self._processor is None:
            raise RuntimeError("Transformers backend requires a processor for input preparation")
        
        # Convert simple string messages to proper format with image placeholders
        formatted_messages = self._format_messages_for_processor(messages, images)
        
        prompt = self._processor.apply_chat_template(formatted_messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=images, return_tensors="pt")
        return inputs
    
    def _format_messages_for_processor(self, messages: List[Dict], images: List[Image.Image]) -> List[Dict]:
        """Convert messages to SmolVLM's expected format with image placeholders.
        
        SmolVLM processor expects messages with content as a list of dicts:
        [{'type': 'image'}, {'type': 'text', 'text': '...'}]
        """
        formatted = []
        image_idx = 0
        
        for msg in messages:
            content = msg.get('content', '')
            role = msg.get('role', 'user')
            
            # If content is already a list, assume it's formatted correctly
            if isinstance(content, list):
                formatted.append(msg)
                continue
            
            # If content is a string, convert to list format with image placeholders
            if isinstance(content, str):
                parts = []
                
                # Add image placeholder for each image (one image per message for now)
                if image_idx < len(images):
                    parts.append({'type': 'image'})
                    image_idx += 1
                
                # Add text content (remove any <image> tokens user might have added)
                text = content.replace('<image>', '').strip()
                if text:
                    parts.append({'type': 'text', 'text': text})
                
                formatted.append({'role': role, 'content': parts})
            else:
                # Unknown format, pass through
                formatted.append(msg)
        
        return formatted

    def generate(self, inputs: Dict[str, Any], max_new_tokens: int | None = None) -> str:
        if max_new_tokens is None:
            max_new_tokens = self._config.max_new_tokens
        generated_ids = self._model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        generated_texts = self._processor.batch_decode(generated_ids, skip_special_tokens=True)
        return generated_texts[0]

    def generate_stream(self, inputs: Dict[str, Any], max_new_tokens: int | None = None) -> Generator[str, None, None]:
        # For transformers we don't have a token stream implemented; yield full result
        full = self.generate(inputs, max_new_tokens=max_new_tokens)
        yield full
