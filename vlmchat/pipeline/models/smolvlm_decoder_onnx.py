"""
ONNX Runtime backend for SmolVLM decoder.

Autoregressive text generation with KV cache.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

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
        
        # Setup execution providers based on model size and device request
        # 500M decoder: CPU only (1.5GB too large for GPU after vision+embed)
        # 256M decoder: Can use CUDA if requested
        if model_size == "500M":
            providers = ['CPUExecutionProvider']
            if device.lower() in ["cuda", "gpu"]:
                logger.info("  Note: 500M decoder forced to CPU (insufficient GPU memory)")
        elif device.lower() in ["cuda", "gpu"]:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        logger.info(f"Loading SmolVLM decoder: {self.engine_path}")
        logger.info(f"  Requested device: {device}")
        logger.info(f"  Providers: {[p[0] if isinstance(p, tuple) else p for p in providers]}")
        logger.info(f"  Layers: {num_hidden_layers}, KV heads: {num_key_value_heads}, head_dim: {head_dim}")
        
        # Create ONNX Runtime session
        self.session = ort.InferenceSession(str(self.engine_path), providers=providers)
        
        # Log actual provider
        actual_provider = self.session.get_providers()[0]
        logger.info(f"  Using: {actual_provider}")
        
        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        logger.info(f"  Inputs: {len(self.input_names)} (embeddings + attention + position + {self.num_hidden_layers * 2} KV pairs)")
        logger.info(f"  Outputs: {len(self.output_names)} (logits + {self.num_hidden_layers * 2} present KV)")
    
    def __del__(self):
        """Cleanup ONNX Runtime session."""
        try:
            if hasattr(self, 'session') and self.session:
                del self.session
                self.session = None
        except Exception:
            pass  # Ignore cleanup errors
    
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
                dtype=np.float32
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
        # Prepare inputs
        ort_inputs = {
            'inputs_embeds': inputs_embeds.astype(np.float32),
            'attention_mask': attention_mask.astype(np.int64),
            'position_ids': position_ids.astype(np.int64),
        }
        
        # Add past key values
        if past_key_values:
            ort_inputs.update(past_key_values)
        
        # Run inference
        outputs = self.session.run(self.output_names, ort_inputs)
        
        # First output is logits
        logits = outputs[0]
        
        # Remaining outputs are present key-values (format: present.X.key, present.X.value)
        present_key_values = {}
        kv_names = [name for name in self.output_names if name.startswith('present.')]
        for i, kv_name in enumerate(kv_names):
            # Convert present.X.key -> past_key_values.X.key
            past_name = kv_name.replace('present.', 'past_key_values.')
            present_key_values[past_name] = outputs[i + 1]
        
        return logits, present_key_values
    
    def __repr__(self) -> str:
        return f"SmolVLMDecoderOnnx(engine={self.engine_path.name}, device={self.device})"
