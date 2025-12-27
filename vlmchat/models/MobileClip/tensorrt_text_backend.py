"""
TensorRT text encoder backend for CLIP.

This module provides the TensorRT implementation for CLIP text encoding.
"""

import logging
import numpy as np
import torch
from typing import List, Optional
from pathlib import Path

# --- Third-party imports ---
try:
    import tensorrt as trt  # type: ignore[import-not-found]
    import pycuda.autoinit  # type: ignore[import-not-found]
    import pycuda.driver as cuda  # type: ignore[import-not-found]
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

try:
    import open_clip  # type: ignore[import-not-found]
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
# --- End third-party imports ---

from .runtime_base import ClipTextRuntimeBase
from utils.config import VLMChatConfig

logger = logging.getLogger(__name__)


class TensorRTTextBackend(ClipTextRuntimeBase):
    """
    CLIP text encoder using TensorRT for accelerated inference.
    """
    
    def __init__(self, config: VLMChatConfig):
        super().__init__(config)
        self.engine = None
        self.context = None
        self.stream = None
        self.tokenizer = None
        self._is_ready = False
        
        # Device buffers
        self.d_input = None
        self.d_output = None
        self.h_input = None
        self.h_output = None
        
        # Tensor info
        self.input_shape = None
        self.output_shape = None
        self.input_dtype = None
        self.output_dtype = None
        self.engine_batch_size = 1
        self.sequence_length = 77
        self.h_input = None
        self.h_output = None
        
        # Tensor info
        self.input_shape = None
        self.output_shape = None
        self.input_dtype = None
        self.output_dtype = None
        
        if not TRT_AVAILABLE:
            logger.error("TensorRTTextBackend Error: TensorRT or PyCUDA not installed.")
            return
        
        if not OPEN_CLIP_AVAILABLE:
            logger.error("TensorRTTextBackend Error: 'open_clip' library not installed (needed for tokenizer).")
            return

        try:
            # Get configuration
            engine_path = getattr(config.model, "clip_text_engine_path", None)
            if not engine_path:
                logger.error("TensorRTTextBackend: clip_text_engine_path not configured")
                return
            
            engine_path = Path(engine_path).expanduser()
            if not engine_path.exists():
                logger.error(f"TensorRTTextBackend: Engine file not found: {engine_path}")
                return
            
            # Load TensorRT engine
            logger.info(f"Loading TensorRT text engine from: {engine_path}")
            trt_logger = trt.Logger(trt.Logger.WARNING)  # type: ignore[possibly-unbound,attr-defined]
            runtime = trt.Runtime(trt_logger)  # type: ignore[possibly-unbound,attr-defined]
            
            with open(engine_path, "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            
            if self.engine is None:
                logger.error("Failed to deserialize TensorRT engine")
                return
            
            # Create execution context
            self.context = self.engine.create_execution_context()
            
            # Get tensor shapes and types
            input_name = "text_input"
            output_name = "text_features"
            
            self.input_shape = tuple(self.context.get_tensor_shape(input_name))
            self.output_shape = tuple(self.context.get_tensor_shape(output_name))
            self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(input_name))  # type: ignore[possibly-unbound]
            self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(output_name))  # type: ignore[possibly-unbound]
            
            # Extract batch size and sequence length from input shape
            self.engine_batch_size = self.input_shape[0] if self.input_shape else 1
            self.sequence_length = self.input_shape[1] if self.input_shape and len(self.input_shape) > 1 else 77
            
            logger.info(f"  Input shape: {self.input_shape}, dtype: {self.input_dtype}")
            logger.info(f"  Output shape: {self.output_shape}, dtype: {self.output_dtype}")
            logger.info(f"  Engine batch size: {self.engine_batch_size}")
            
            # Allocate buffers
            input_size = int(np.prod(self.input_shape))
            output_size = int(np.prod(self.output_shape))
            
            self.h_input = cuda.pagelocked_empty(input_size, self.input_dtype)  # type: ignore[possibly-unbound,attr-defined]
            self.h_output = cuda.pagelocked_empty(output_size, self.output_dtype)  # type: ignore[possibly-unbound,attr-defined]
            
            self.d_input = cuda.mem_alloc(self.h_input.nbytes)  # type: ignore[possibly-unbound,attr-defined]
            self.d_output = cuda.mem_alloc(self.h_output.nbytes)  # type: ignore[possibly-unbound,attr-defined]
            
            # Bind tensors
            self.context.set_tensor_address(input_name, int(self.d_input))
            self.context.set_tensor_address(output_name, int(self.d_output))
            
            # Create CUDA stream
            self.stream = cuda.Stream()  # type: ignore[possibly-unbound,attr-defined]
            
            # Load tokenizer from OpenCLIP
            model_name = getattr(config.model, "clip_model_name", "MobileCLIP2-S0")
            self.tokenizer = open_clip.get_tokenizer(model_name)  # type: ignore[possibly-unbound]
            
            self._is_ready = True
            logger.info("TensorRTTextBackend loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT text backend: {e}", exc_info=True)
            self._is_ready = False
    
    @property
    def is_available(self) -> bool:
        """Returns True if the backend loaded successfully."""
        return self._is_ready and self.engine is not None
    
    @property
    def max_batch_size(self) -> int:
        """Returns the maximum batch size supported by the TensorRT engine."""
        return self.engine_batch_size
    
    def encode_text(self, text_prompts: List[str]) -> torch.Tensor:
        """Encodes a list of text strings using TensorRT."""
        if not self.is_available or self.tokenizer is None:
            raise RuntimeError("TensorRTTextBackend is not ready.")
        
        # Tokenize text
        tokenized = self.tokenizer(text_prompts)
        
        # Convert to numpy for TensorRT
        if isinstance(tokenized, torch.Tensor):
            np_input = tokenized.cpu().numpy()
        else:
            np_input = np.array(tokenized)
        
        # Get batch size (number of prompts)
        batch_size = len(text_prompts)
        
        # Ensure correct shape: (batch_size, sequence_length)
        if np_input.ndim == 1:
            # Single flattened array - reshape to (batch_size, tokens_per_prompt)
            np_input = np_input.reshape(batch_size, self.sequence_length)
        elif np_input.shape[0] != batch_size:
            # Wrong batch size - reshape
            np_input = np_input.reshape(batch_size, self.sequence_length)
        
        # Process prompts in batches matching the engine's batch size
        all_features = []
        for batch_start in range(0, batch_size, self.engine_batch_size):
            batch_end = min(batch_start + self.engine_batch_size, batch_size)
            batch_inputs = np_input[batch_start:batch_end]
            
            # Pad batch if needed (when last batch is smaller than engine batch size)
            actual_batch_size = batch_inputs.shape[0]
            if actual_batch_size < self.engine_batch_size:
                padding = np.zeros((self.engine_batch_size - actual_batch_size, self.sequence_length), 
                                   dtype=self.input_dtype)  # type: ignore[attr-defined]
                batch_inputs = np.vstack([batch_inputs, padding])
            
            # Convert to correct dtype and reshape to engine input shape
            batch_inputs = batch_inputs.astype(self.input_dtype, copy=False)  # type: ignore[attr-defined]
            
            # Copy to host buffer
            self.h_input[:] = batch_inputs.ravel()  # type: ignore[index]
            
            # Transfer to device and execute
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)  # type: ignore[possibly-unbound,attr-defined]
            self.context.execute_async_v3(stream_handle=self.stream.handle)  # type: ignore[attr-defined]
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)  # type: ignore[possibly-unbound,attr-defined]
            self.stream.synchronize()  # type: ignore[attr-defined]
            
            # Convert output to numpy array
            output = np.array(self.h_output).reshape(self.output_shape)  # type: ignore[arg-type]
            if output.dtype == np.float16:
                output = output.astype(np.float32)
            
            # Only keep the actual batch size (discard padding)
            output = output[:actual_batch_size]
            all_features.append(output)
        
        # Stack all features
        text_features = torch.from_numpy(np.vstack(all_features))
        
        # Normalize features (L2 normalization)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        return text_features
    
    def __del__(self):
        """Cleanup CUDA resources."""
        if self.d_input:
            self.d_input.free()
        if self.d_output:
            self.d_output.free()
