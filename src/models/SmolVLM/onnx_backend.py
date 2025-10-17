"""
ONNX backend implementing the BackendBase interface.

This backend is independent and will import any HF components it needs
internally. It mirrors the same public methods as the Transformers backend
so callers can switch backends transparently.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Generator, Optional

import numpy as np
import onnxruntime
from transformers import AutoProcessor, AutoConfig
from PIL import Image

from models.SmolVLM.runtime_base import RuntimeBase
from models.SmolVLM.model_config import ModelConfig
from utils.onnx_utils import get_onnx_file_paths, setup_onnx_environment
from config import get_config

from utils.metrics_collector import Collector, null_collector

logger = logging.getLogger(__name__)


class OnnxBackend(RuntimeBase):
    def __init__(self, config: ModelConfig, collector: Optional[Collector] = null_collector()):
        self._config = config
        self._use_onnx = True
        self._processor = None
        self._hf_config = None
        self._collector = collector
        self._eos_token_id = []

        try:
            # Try to load processor/config for input preparation
            try:
                self._hf_config = AutoConfig.from_pretrained(self._config.model_path)
                self._processor = AutoProcessor.from_pretrained(self._config.model_path)
                self._tokenizer = self._processor.tokenizer
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
                # Ensure eos token ids are stored as a list so callers can
                # safely append or iterate over them. HF configs may provide
                # a single int or a sequence.
                cfg_eos = text_config.eos_token_id
                if cfg_eos is None:
                    self._eos_token_id = []
                elif isinstance(cfg_eos, (list, tuple)):
                    self._eos_token_id = list(cfg_eos)
                else:
                    # cast scalars to int and wrap in a list
                    try:
                        self._eos_token_id = [int(cfg_eos)]
                    except Exception:
                        # fallback to empty list on unexpected types
                        self._eos_token_id = []

                self._image_token_id = self._hf_config.image_token_id
            eos = self._analyze_special_tokens()
            if eos is not None and eos != 0:
                logger.info(f"Adding EOS token: {eos}")
                self._eos_token_id.append(eos)

            logger.info("ONNX sessions loaded")

        except Exception as e:
            logger.warning(f"Failed to initialize OnnxBackend: {e}")
            self._use_onnx = False

    @property
    def is_available(self) -> bool:
        return self._use_onnx

    def _analyze_special_tokens(self):
        """
        Analyze and log special token information for debugging.

        Examines special tokens used by the model to ensure they are properly
        configured in the tokenizer vocabulary. Logs token information for
        troubleshooting tokenization issues.
        """
        special_token = self._config.special_tokens["end_of_utterance"]
        is_in_vocab = special_token in self._tokenizer.vocab
        token_id = self._tokenizer.convert_tokens_to_ids(special_token)

        logger.info(f"Special token '{special_token}' in vocabulary: {is_in_vocab}")
        logger.info(f"Token ID for '{special_token}': {token_id}")
        logger.info(f"EOS token: {self._tokenizer.eos_token}")
        logger.info(f"EOS token ID: {self._tokenizer.eos_token_id}")
        return token_id

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
            with self._collector.duration_timer("smolVLM-onnx", {"embeds": None}):
                inputs_embeds = self._embed_session.run(None, {'input_ids': input_ids})[0]

            if image_features is None:
                with self._collector.duration_timer("smolVLM-onnx", {"vision-encoder": None}):
                    image_features = self._vision_session.run(
                        ['image_features'],
                        {
                            'pixel_values': inputs['pixel_values'],
                            'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
                        }
                    )[0]

                inputs_embeds[inputs['input_ids'] == self._image_token_id] = \
                    image_features.reshape(-1, image_features.shape[-1])

            with self._collector.duration_timer("smolVLM-onnx", {"decoder": None}):
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
