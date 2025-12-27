"""
ONNX backend implementing the BackendBase interface.

This backend is independent and will import any HF components it needs
internally. It mirrors the same public methods as the Transformers backend
so callers can switch backends transparently.
"""
from __future__ import annotations

import logging
import traceback
from typing import Any, Dict, List, Generator, Optional

import numpy as np
import onnxruntime
from onnxruntime import OrtValue
from transformers import AutoProcessor, AutoConfig
from PIL import Image

from models.SmolVLM.runtime_base import SmolVLMRuntimeBase
from models.SmolVLM.model_config import ModelConfig
from utils.onnx_utils import get_onnx_file_paths, setup_onnx_environment
from utils.config import VLMChatConfig

from metrics.metrics_collector import Collector, null_collector

logger = logging.getLogger(__name__)


class OnnxBackend(SmolVLMRuntimeBase):
    def __init__(self, config: VLMChatConfig, collector: Collector = null_collector()):
        super().__init__(config)
        self._model_config = ModelConfig(config)
        self._use_onnx = True
        self._processor = None
        self._hf_config = None
        self._collector = collector
        self._eos_token_id = []
        self._device: str = 'cpu'
        self._use_io_binding = False

        try:
            # Try to load processor/config for input preparation
            try:
                self._hf_config = AutoConfig.from_pretrained(self._model_config.model_path)
                self._processor = AutoProcessor.from_pretrained(self._model_config.model_path)
                self._tokenizer = self._processor.tokenizer
            except Exception as e:
                # processor is optional for ONNX if inputs are made externally
                logger.warning(f"Failed to load processor/tokenizer: {e}")
                self._hf_config = None
                self._processor = None
                self._tokenizer = None

            if not setup_onnx_environment(config):
                self._use_onnx = False
                return
            self._collector.register_timeseries("smolVLM-onnx", ["vision-encoder","embeds", "generate"], ttl_seconds=600)
            onnx_model_path = config.model.get_onnx_model_path()
            onnx_files = get_onnx_file_paths(onnx_model_path)

            # Use TensorRT for vision/embed, CPU for decoder (has compatibility issues)
            gpu_providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
            cpu_only = ['CPUExecutionProvider']
            
            self._vision_session = onnxruntime.InferenceSession(str(onnx_files["vision_encoder"]), providers=gpu_providers)
            self._embed_session = onnxruntime.InferenceSession(str(onnx_files["embed_tokens"]), providers=gpu_providers)
            self._decoder_session = onnxruntime.InferenceSession(str(onnx_files["decoder"]), providers=cpu_only)
            
            # Detect GPU provider
            providers = self._vision_session.get_providers()
            if 'TensorrtExecutionProvider' in providers:
                self._device = 'cuda'
                self._use_io_binding = False  # Disable IO binding for now (testing CPU path with GPU models)
                logger.info(f"GPU detected (TensorRT), using CPU generation path")
                logger.info(f"Vision: {self._vision_session.get_providers()[0]}")
                logger.info(f"Embed: {self._embed_session.get_providers()[0]}")
                logger.info(f"Decoder: {self._decoder_session.get_providers()[0]}")
            else:
                self._device = 'cpu'
                self._use_io_binding = False
                logger.info("Running on CPU")

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
            
            # Analyze special tokens if tokenizer is available
            if self._tokenizer is not None:
                eos = self._analyze_special_tokens()
                if eos is not None and eos != 0:
                    logger.info(f"Adding tokenizer EOS token: {eos}")
                    self._eos_token_id.append(eos)
            
            logger.info(f"EOS tokens configured: {self._eos_token_id}")
            logger.info("ONNX sessions loaded")

        except Exception as e:
            logger.warning(f"Failed to initialize OnnxBackend: {e}")
            traceback.print_exc()
            self._use_onnx = False

    @property
    def native_image_format(self) -> str:
        """SmolVLM ONNX backend expects PIL images."""
        return "pil"

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
        if self._tokenizer is None:
            logger.warning("Tokenizer not available, skipping special token analysis")
            return None
            
        special_token = self._model_config.special_tokens["end_of_utterance"]
        is_in_vocab = special_token in self._tokenizer.vocab
        token_id = self._tokenizer.convert_tokens_to_ids(special_token)

        logger.info(f"Special token '{special_token}' in vocabulary: {is_in_vocab}")
        logger.info(f"Token ID for '{special_token}': {token_id}")
        logger.info(f"EOS token: {self._tokenizer.eos_token}")
        logger.info(f"EOS token ID: {self._tokenizer.eos_token_id}")
        return token_id
    
    def _create_ortvalue_from_numpy(self, array: np.ndarray, device: str = 'cuda') -> OrtValue:
        """
        Create an OrtValue tensor on the specified device from a NumPy array.
        
        Args:
            array: NumPy array to convert
            device: Device to place tensor on ('cuda' or 'cpu')
            
        Returns:
            OrtValue tensor on the specified device
        """
        return OrtValue.ortvalue_from_numpy(array, device, 0)
    
    def _ortvalue_to_numpy(self, ortvalue: OrtValue) -> np.ndarray:
        """
        Convert an OrtValue tensor to a NumPy array on CPU.
        
        Args:
            ortvalue: OrtValue tensor (may be on GPU)
            
        Returns:
            NumPy array on CPU
        """
        return ortvalue.numpy()
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
            max_new_tokens = self._model_config.max_new_tokens

        # Use optimized GPU path if available
        if self._use_io_binding:
            yield from self._generate_stream_gpu(inputs, max_new_tokens)
        else:
            yield from self._generate_stream_cpu(inputs, max_new_tokens)
    
    def _generate_stream_cpu(self, inputs: Dict[str, Any], max_new_tokens: int) -> Generator[str, None, None]:
        """Original CPU-based generation (fallback for non-GPU systems)."""
        batch_size = inputs['input_ids'].shape[0]
        past_key_values = self._initialize_past_key_values(batch_size)
        image_features = None
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = np.cumsum(inputs['attention_mask'], axis=-1)

        for i in range(max_new_tokens):
            with self._collector.duration_timer("smolVLM-onnx", {"embeds": ""}):
                inputs_embeds = self._embed_session.run(None, {'input_ids': input_ids})[0]

            if image_features is None:
                with self._collector.duration_timer("smolVLM-onnx", {"vision-encoder": ""}):
                    image_features = self._vision_session.run(
                        ['image_features'],
                        {
                            'pixel_values': inputs['pixel_values'],
                            'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
                        }
                    )[0]

                # type: ignore[index] for array assignment
                inputs_embeds[inputs['input_ids'] == self._image_token_id] = image_features.reshape(-1, image_features.shape[-1])  # type: ignore[union-attr]

            with self._collector.duration_timer("smolVLM-onnx", {"decoder": ""}):
                logits, *present_key_values = self._decoder_session.run(None, dict(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **past_key_values,
                ))

            input_ids = logits[:, -1].argmax(-1, keepdims=True)  # type: ignore[call-overload,index]
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1

            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]  # type: ignore[assignment]

            # Yield the decoded token
            yield self._processor.decode(input_ids[0])  # type: ignore[union-attr]

            # Check if generated token is an EOS token
            token_id = int(input_ids[0, 0])
            if self._eos_token_id and token_id in self._eos_token_id:
                break
    
    def _generate_stream_gpu(self, inputs: Dict[str, Any], max_new_tokens: int) -> Generator[str, None, None]:
        """Optimized GPU generation using IO binding to keep tensors on GPU."""
        batch_size = inputs['input_ids'].shape[0]
        past_key_values = self._initialize_past_key_values(batch_size)
        image_features_gpu = None
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = np.cumsum(inputs['attention_mask'], axis=-1)

        for i in range(max_new_tokens):
            # Embed tokens - run on GPU
            with self._collector.duration_timer("smolVLM-onnx", {"embeds": ""}):
                io_binding = self._embed_session.io_binding()
                io_binding.bind_cpu_input('input_ids', input_ids)
                io_binding.bind_output('inputs_embeds', self._device)
                self._embed_session.run_with_iobinding(io_binding)
                inputs_embeds_gpu = io_binding.get_outputs()[0]

            # Vision encoder - only run once, keep features on GPU
            if image_features_gpu is None:
                with self._collector.duration_timer("smolVLM-onnx", {"vision-encoder": ""}):
                    io_binding = self._vision_session.io_binding()
                    io_binding.bind_cpu_input('pixel_values', inputs['pixel_values'])
                    io_binding.bind_cpu_input('pixel_attention_mask', inputs['pixel_attention_mask'].astype(np.bool_))
                    io_binding.bind_output('image_features', self._device)
                    self._vision_session.run_with_iobinding(io_binding)
                    image_features_gpu = io_binding.get_outputs()[0]

                # Inject image features into embeddings (need to do on CPU for indexing)
                inputs_embeds_cpu = self._ortvalue_to_numpy(inputs_embeds_gpu)
                image_features_cpu = self._ortvalue_to_numpy(image_features_gpu)
                inputs_embeds_cpu[inputs['input_ids'] == self._image_token_id] = image_features_cpu.reshape(-1, image_features_cpu.shape[-1])
                inputs_embeds_gpu = self._create_ortvalue_from_numpy(inputs_embeds_cpu)

            # Decoder - keep KV cache on GPU
            with self._collector.duration_timer("smolVLM-onnx", {"decoder": ""}):
                io_binding = self._decoder_session.io_binding()
                io_binding.bind_ortvalue_input('inputs_embeds', inputs_embeds_gpu)
                io_binding.bind_cpu_input('attention_mask', attention_mask)
                io_binding.bind_cpu_input('position_ids', position_ids)
                
                # Bind past key values
                for key, value in past_key_values.items():
                    if isinstance(value, OrtValue):
                        io_binding.bind_ortvalue_input(key, value)
                    else:
                        io_binding.bind_cpu_input(key, value)
                
                # Bind outputs - logits to CPU for argmax, KV cache stays on GPU
                io_binding.bind_output('logits', 'cpu')
                for layer in range(self._num_hidden_layers):
                    for kv in ('key', 'value'):
                        io_binding.bind_output(f'present.{layer}.{kv}', self._device)
                
                self._decoder_session.run_with_iobinding(io_binding)
                outputs = io_binding.get_outputs()
                logits = self._ortvalue_to_numpy(outputs[0])
                
                # Update KV cache (keep on GPU)
                for j, key in enumerate(past_key_values.keys()):
                    past_key_values[key] = outputs[j + 1]  # type: ignore[assignment]

            input_ids = logits[:, -1].argmax(-1, keepdims=True)  # type: ignore[call-overload,index]
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1

            # Yield the decoded token
            yield self._processor.decode(input_ids[0])  # type: ignore[union-attr]

            # Check if generated token is an EOS token
            token_id = int(input_ids[0, 0])
            if self._eos_token_id and token_id in self._eos_token_id:
                break

    def generate(self, inputs: Dict[str, Any], max_new_tokens: int | None = None) -> str:
        # Collect stream into a full string
        parts = []
        for token in self.generate_stream(inputs, max_new_tokens=max_new_tokens):
            parts.append(token)
        parts.append('\n')
        return ''.join(parts)
