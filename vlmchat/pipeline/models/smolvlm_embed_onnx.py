"""
ONNX Runtime backend for SmolVLM token embeddings.

Converts token IDs to embeddings.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Union

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

# Check for TensorRT availability
try:
    import tensorrt as trt
    TRT_AVAILABLE = 'TensorrtExecutionProvider' in ort.get_available_providers()
except ImportError:
    TRT_AVAILABLE = False


class SmolVLMEmbedOnnx:
    """
    ONNX Runtime backend for SmolVLM token embeddings.
    
    Converts input token IDs into embeddings for the language model.
    """
    
    def __init__(self, 
                 engine_path: str,
                 device: str = "cuda"):
        """
        Initialize SmolVLM embedding layer.
        
        Args:
            engine_path: Path to embed_tokens.onnx file
            device: Device for inference ('cuda' or 'cpu')
        """
        self.engine_path = Path(engine_path)
        self.device = device
        
        if not self.engine_path.exists():
            raise FileNotFoundError(f"Embedding engine not found: {engine_path}")
        
# Set up providers based on device (simple format like working onnx_backend.py)
        if device == "cuda" and TRT_AVAILABLE:
            providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
        elif device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        logger.info(f"Loading SmolVLM embedding layer: {self.engine_path}")
        logger.info(f"  Requested device: {device}")
        logger.info(f"  Providers: {providers}")
        
        # Create ONNX Runtime session (simple format, no SessionOptions or provider_options)
        self.session = ort.InferenceSession(str(self.engine_path), providers=providers)
        
        # Log actual provider
        actual_provider = self.session.get_providers()[0]
        logger.info(f"  Using: {actual_provider}")
        
        # Get input/output info
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        logger.info(f"  Inputs: {self.input_names}")
        logger.info(f"  Outputs: {self.output_names}")
    
    def __del__(self):
        """Cleanup ONNX Runtime session."""
        try:
            if hasattr(self, 'session') and self.session:
                del self.session
                self.session = None
        except Exception:
            pass  # Ignore cleanup errors
    
    def embed(self, input_ids: np.ndarray) -> np.ndarray:
        """
        Convert token IDs to embeddings.
        
        Args:
            input_ids: Token IDs, shape (batch, sequence_length)
            
        Returns:
            Token embeddings, shape (batch, sequence_length, hidden_dim)
        """
        # Prepare inputs
        ort_inputs = {
            'input_ids': input_ids.astype(np.int64)
        }
        
        # Run inference
        outputs = self.session.run(self.output_names, ort_inputs)
        
        # Return embeddings
        return outputs[0]
    
    def __repr__(self) -> str:
        return f"SmolVLMEmbedOnnx(engine={self.engine_path.name}, device={self.device})"
