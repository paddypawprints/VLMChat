"""
ONNX backend implementing the BackendBase interface.

This backend is independent and will import any HF components it needs
internally. It mirrors the same public methods as the Transformers backend
so callers can switch backends transparently.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Generator

import numpy as np
import onnxruntime
from transformers import AutoProcessor, AutoConfig
from PIL import Image

from models.SmolVLM.backend_base import BackendBase
from models.SmolVLM.model_config import ModelConfig
from utils.onnx_utils import get_onnx_file_paths, setup_onnx_environment
from config import get_config

logger = logging.getLogger(__name__)


class OnnxBackend(BackendBase):
    def __init__(self, config: ModelConfig):
        self._config = config
        self._use_onnx = True
        self._processor = None
        self._hf_config = None

        try:
            # Try to load processor/config for input preparation
            try:
                self._hf_config = AutoConfig.from_pretrained(self._config.model_path)
                self._processor = AutoProcessor.from_pretrained(self._config.model_path)
            except Exception:
                # processor is optional for ONNX if inputs are made externally
                self._hf_config = None
                self._processor = None

            global_config = get_config()
            if not setup_onnx_environment(global_config):
                self._use_onnx = False
                return

            onnx_model_path = global_config.model.get_onnx_model_path()
            onnx_files = get_onnx_file_paths(onnx_model_path)

            self._vision_session = onnxruntime.InferenceSession(str(onnx_files["vision_encoder"]))
            self._embed_session = onnxruntime.InferenceSession(str(onnx_files["embed_tokens"]))
            self._decoder_session = onnxruntime.InferenceSession(str(onnx_files["decoder"]))

            if self._hf_config is not None:
                text_config = self._hf_config.text_config
                self._num_key_value_heads = text_config.num_key_value_heads
                self._head_dim = text_config.head_dim
                self._num_hidden_layers = text_config.num_hidden_layers
                self._eos_token_id = text_config.eos_token_id
                self._image_token_id = self._hf_config.image_token_id

            logger.info("ONNX sessions loaded")

        except Exception as e:
            logger.warning(f"Failed to initialize OnnxBackend: {e}")
            self._use_onnx = False

    @property
    def is_available(self) -> bool:
        return self._use_onnx

    def prepare_inputs(self, messages: List[Dict], images: List[Image.Image]) -> Dict[str, Any]:
        if self._processor is None:
            raise RuntimeError("ONNX backend requires a processor for input preparation")
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._processor(text=prompt, images=images, return_tensors="np")
        return inputs

    def _initialize_past_key_values(self, batch_size: int) -> Dict[str, np.ndarray]:
        return {
            f'past_key_values.{layer}.{kv}': np.zeros(
                [batch_size, self._num_key_value_heads, 0, self._head_dim],
                dtype=np.float32
            )
            for layer in range(self._num_hidden_layers)
            for kv in ('key', 'value')
        }

    def generate_stream(self, inputs: Dict[str, Any], max_new_tokens: int | None = None) -> Generator[str, None, None]:
        if not self._use_onnx:
            raise RuntimeError("ONNX backend not available")

        if max_new_tokens is None:
            max_new_tokens = self._config.max_new_tokens

        batch_size = inputs['input_ids'].shape[0]
        past_key_values = self._initialize_past_key_values(batch_size)
        image_features = None
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = np.cumsum(inputs['attention_mask'], axis=-1)

        for i in range(max_new_tokens):
            inputs_embeds = self._embed_session.run(None, {'input_ids': input_ids})[0]

            if image_features is None:
                image_features = self._vision_session.run(
                    ['image_features'],
                    {
                        'pixel_values': inputs['pixel_values'],
                        'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
                    }
                )[0]

                inputs_embeds[inputs['input_ids'] == self._image_token_id] = \
                    image_features.reshape(-1, image_features.shape[-1])

            logits, *present_key_values = self._decoder_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **past_key_values,
            ))

            input_ids = logits[:, -1].argmax(-1, keepdims=True)
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1

            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

            if (input_ids == self._eos_token_id).any():
                break

            yield self._processor.decode(input_ids[0])

    def generate(self, inputs: Dict[str, Any], max_new_tokens: int | None = None) -> str:
        # Collect stream into a full string
        parts = []
        for token in self.generate_stream(inputs, max_new_tokens=max_new_tokens):
            parts.append(token)
        parts.append('\n')
        return ''.join(parts)
