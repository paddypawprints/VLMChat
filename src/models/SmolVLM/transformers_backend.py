"""
Transformers backend implementing the BackendBase interface.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Generator, Optional

import torch
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText

from PIL import Image

from models.SmolVLM.runtime_base import RuntimeBase
from models.SmolVLM.model_config import ModelConfig

from utils.metrics_collector import Collector, null_collector

logger = logging.getLogger(__name__)


class TransformersBackend(RuntimeBase):
    def __init__(self, model_path: str, config: ModelConfig,  collector: Optional[Collector] = null_collector()):
        self._model_path = model_path
        self._config = config
        self._available = True
        self._collector = collector

        try:
            with self._collector.duration_timer("SmolVLM-transformers", {"initialize" : None}):
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
    def is_available(self) -> bool:
        return self._available

    def prepare_inputs(self, messages: List[Dict], images: List[Image.Image]) -> Dict[str, Any]:
#        inputs = self._processor.apply_chat_template(
#            messages,
#            add_generation_prompt=True,
#            tokenize=True,
#            return_dict=True,
#            return_tensors="pt",
#        ).to(self._model.device, dtype=torch.bfloat16)
        if self._processor is None:
            raise RuntimeError("Transformers backend requires a processor for input preparation")
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=images, return_tensors="pt")
        return inputs

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
