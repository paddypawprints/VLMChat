"""
ONNX Runtime backend for SmolVLM decoder.

Autoregressive text generation with KV cache.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class SmolVLMDecoderOnnx:
    """
    ONNX Runtime backend for SmolVLM decoder.

    Performs autoregressive generation with past key-value caching.
    Note: Decoder typically runs on CPU due to ONNX compatibility issues.
    """

    def __init__(self,
                 engine_path: str,
                 num_hidden_layers: int = 32,
                 num_key_value_heads: int = 8,
                 head_dim: int = 64,
                 device: str = "cpu",
                 model_size: str = "256M"):
        """
        Initialize SmolVLM decoder.

        Args:
            engine_path: Path to decoder_model_merged.onnx file
            num_hidden_layers: Number of transformer layers
            num_key_value_heads: Number of key-value attention heads
            head_dim: Dimension per attention head
            device: Device for inference ('cpu' recommended)
            model_size: Model size ('256M' or '500M') for GPU memory decisions
        """
        self.engine_path = Path(engine_path)
        self.device = device
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim

        if not self.engine_path.exists():
            raise FileNotFoundError(f"Decoder not found: {engine_path}")

        # The decoder uses a merged ONNX model (handles both the prefill step where
        # past_key_values has seq_len=0 and subsequent steps where seq_len>0).
        # CoreML EP cannot execute kernels with zero-element tensors, so it crashes
        # on the very first decode step.  CPU handles dynamic/zero shapes correctly
        # and is fast enough for token-by-token prediction.
        providers = ['CPUExecutionProvider']
        if device.lower() in ["cuda", "gpu"]:
            logger.info("  Note: decoder forced to CPU (CoreML EP rejects zero-dim KV cache on first step)")

        logger.info(f"Loading SmolVLM decoder: {self.engine_path}")
        logger.info(f"  Requested device: {device}")
        logger.info(f"  Providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
        logger.info(f"  Layers: {num_hidden_layers}, KV heads: {num_key_value_heads}, head_dim: {head_dim}")

        self.session = ort.InferenceSession(str(self.engine_path), providers=providers)

        actual_provider = self.session.get_providers()[0]
        logger.info(f"  Using: {actual_provider}")

        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

        logger.info(
            f"  Inputs: {len(self.input_names)} "
            f"(embeddings + attention + position + {self.num_hidden_layers * 2} KV pairs)"
        )
        logger.info(
            f"  Outputs: {len(self.output_names)} "
            f"(logits + {self.num_hidden_layers * 2} present KV)"
        )

    def __del__(self):
        """Cleanup ONNX Runtime session."""
        try:
            if hasattr(self, 'session') and self.session:
                del self.session
                self.session = None
        except Exception:
            pass

    def initialize_past_key_values(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Initialize empty past key-value cache.

        Args:
            batch_size: Batch size for generation

        Returns:
            Dictionary of empty KV cache arrays
        """
        return {
            f'past_key_values.{layer}.{kv}': np.zeros(
                [batch_size, self.num_key_value_heads, 0, self.head_dim],
                dtype=np.float32,
            )
            for layer in range(self.num_hidden_layers)
            for kv in ('key', 'value')
        }

    def decode(self,
               inputs_embeds: np.ndarray,
               attention_mask: np.ndarray,
               position_ids: np.ndarray,
               past_key_values: Dict[str, np.ndarray]) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Single decoder step with KV cache.

        Args:
            inputs_embeds: Input embeddings, shape (batch, seq_len, hidden_dim)
            attention_mask: Attention mask, shape (batch, seq_len)
            position_ids: Position IDs, shape (batch, seq_len)
            past_key_values: Past key-value cache from previous steps

        Returns:
            Tuple of:
                - Logits: shape (batch, seq_len, vocab_size)
                - Present key-values: Updated KV cache for next step
        """
        ort_inputs = {
            'inputs_embeds': inputs_embeds.astype(np.float32),
            'attention_mask': attention_mask.astype(np.int64),
            'position_ids': position_ids.astype(np.int64),
        }
        if past_key_values:
            ort_inputs.update(past_key_values)

        outputs = self.session.run(self.output_names, ort_inputs)

        logits = outputs[0]

        # Convert present.X.key → past_key_values.X.key for next step
        present_key_values = {}
        kv_names = [name for name in self.output_names if name.startswith('present.')]
        for i, kv_name in enumerate(kv_names):
            past_name = kv_name.replace('present.', 'past_key_values.')
            present_key_values[past_name] = outputs[i + 1]

        return logits, present_key_values

    def __repr__(self) -> str:
        return f"SmolVLMDecoderOnnx(engine={self.engine_path.name}, device={self.device})"
