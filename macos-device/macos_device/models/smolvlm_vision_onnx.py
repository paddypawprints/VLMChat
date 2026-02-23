"""
ONNX Runtime backend for SmolVLM vision encoder.

Encodes images into visual features for the language model.
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


class SmolVLMVisionOnnx:
    """
    ONNX Runtime backend for SmolVLM vision encoder.

    Processes images through the vision encoder to produce image features.
    """

    def __init__(self,
                 engine_path: str,
                 device: str = "cuda"):
        """
        Initialize SmolVLM vision encoder.

        Args:
            engine_path: Path to vision_encoder.onnx file
            device: Device for inference ('cuda' or 'cpu')
        """
        self.engine_path = Path(engine_path)
        self.device = device

        if not self.engine_path.exists():
            raise FileNotFoundError(f"Vision encoder not found: {engine_path}")

        # Set up providers based on device.
        # On macOS, prefer CoreML for acceleration on Apple Silicon.
        if device == "cuda" and TRT_AVAILABLE:
            providers = ['TensorrtExecutionProvider', 'CPUExecutionProvider']
        elif device == "cuda":
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        elif 'CoreMLExecutionProvider' in ort.get_available_providers():
            providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']

        logger.info(f"Loading SmolVLM vision encoder: {self.engine_path}")
        logger.info(f"  Requested device: {device}")
        logger.info(f"  Providers: {providers}")

        self.session = ort.InferenceSession(str(self.engine_path), providers=providers)

        actual_provider = self.session.get_providers()[0]
        logger.info(f"  Using: {actual_provider}")

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
            pass

    def encode(self,
               pixel_values: np.ndarray,
               pixel_attention_mask: np.ndarray) -> np.ndarray:
        """
        Encode images to visual features.

        Args:
            pixel_values: Preprocessed pixel values, shape (batch, channels, height, width)
            pixel_attention_mask: Attention mask for pixels, shape (batch, height, width)

        Returns:
            Image features array, shape (batch, num_patches, hidden_dim)
        """
        ort_inputs = {
            'pixel_values': pixel_values.astype(np.float32),
            'pixel_attention_mask': pixel_attention_mask.astype(np.bool_),
        }
        outputs = self.session.run(self.output_names, ort_inputs)
        return outputs[0]

    def __repr__(self) -> str:
        return f"SmolVLMVisionOnnx(engine={self.engine_path.name}, device={self.device})"
