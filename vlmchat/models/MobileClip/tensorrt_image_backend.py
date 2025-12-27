"""
TensorRT image encoder backend for CLIP.

This module provides the TensorRT implementation for CLIP image encoding.
"""

import logging
import numpy as np
import torch
from typing import Optional
from PIL import Image
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
    from mobileclip.modules.common.mobileone import reparameterize_model  # type: ignore[import-not-found]
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
# --- End third-party imports ---

from .runtime_base import ClipImageRuntimeBase
from utils.config import VLMChatConfig

logger = logging.getLogger(__name__)


class TensorRTImageBackend(ClipImageRuntimeBase):
    """
    CLIP image encoder using TensorRT for accelerated inference.
    """
    
    def __init__(self, config: VLMChatConfig):
        super().__init__(config)
        self.engine = None
        self.context = None
        self.stream = None
        self.preprocess = None
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
        
        if not TRT_AVAILABLE:
            logger.error("TensorRTImageBackend Error: TensorRT or PyCUDA not installed.")
            return
        
        if not OPEN_CLIP_AVAILABLE:
            logger.error("TensorRTImageBackend Error: 'open_clip' library not installed (needed for preprocessing).")
            return

        try:
            # Get configuration
            engine_path = getattr(config.model, "clip_image_engine_path", None)
            if not engine_path:
                logger.error("TensorRTImageBackend: clip_image_engine_path not configured")
                return
            
            engine_path = Path(engine_path).expanduser()
            if not engine_path.exists():
                logger.error(f"TensorRTImageBackend: Engine file not found: {engine_path}")
                return
            
            # Load TensorRT engine
            logger.info(f"Loading TensorRT image engine from: {engine_path}")
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
            input_name = "image_input"
            output_name = "image_features"
            
            self.input_shape = tuple(self.context.get_tensor_shape(input_name))
            self.output_shape = tuple(self.context.get_tensor_shape(output_name))
            self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(input_name))  # type: ignore[possibly-unbound]
            self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(output_name))  # type: ignore[possibly-unbound]
            
            # Extract batch size from input shape
            self.engine_batch_size = self.input_shape[0] if self.input_shape else 1
            
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
            
            # Load preprocessing from OpenCLIP (for consistency)
            model_name = getattr(config.model, "clip_model_name", "MobileCLIP2-S0")
            model_kwargs = getattr(config.model, "clip_model_kwargs", None) or {}
            
            # Apply MobileCLIP-specific normalization
            if "MobileCLIP" in model_name:
                if 'image_mean' not in model_kwargs and 'image_std' not in model_kwargs:
                    if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
                        model_kwargs["image_mean"] = (0, 0, 0)
                        model_kwargs["image_std"] = (1, 1, 1)
            
            _, _, self.preprocess = open_clip.create_model_and_transforms(  # type: ignore[possibly-unbound]
                model_name,
                pretrained=None,  # Don't load weights, just get transforms
                **model_kwargs
            )
            
            self._is_ready = True
            logger.info("TensorRTImageBackend loaded successfully.")
            
        except Exception as e:
            logger.error(f"Failed to load TensorRT image backend: {e}", exc_info=True)
            self._is_ready = False
    
    @property
    def native_image_format(self) -> str:
        """TensorRT backend prefers GPU tensors for optimal performance."""
        return "torch_gpu"
    
    @property
    def is_available(self) -> bool:
        """Returns True if the backend loaded successfully."""
        return self._is_ready and self.engine is not None
    
    @property
    def max_batch_size(self) -> int:
        """Returns the maximum batch size supported by the TensorRT engine."""
        return self.engine_batch_size
    
    def encode_image(self, image: Image.Image) -> torch.Tensor:
        """
        Encodes a single image using TensorRT.
        
        Accepts either PIL Image or GPU tensor in native format.
        For optimal performance, provide preprocessed GPU tensor.
        
        Args:
            image: PIL Image or torch.Tensor (GPU)
            
        Returns:
            Normalized feature tensor
        """
        if not self.is_available or self.preprocess is None:
            raise RuntimeError("TensorRTImageBackend is not ready.")
        
        # Fast path: already a GPU tensor in correct format
        if isinstance(image, torch.Tensor) and image.is_cuda:
            # Assume tensor is already preprocessed (C, H, W) format
            torch_input = image.unsqueeze(0) if image.ndim == 3 else image
            # Convert directly from GPU to numpy for TensorRT (single GPU→CPU transfer)
            assert self.input_shape is not None, "TensorRT input shape not initialized"
            np_input = torch_input.cpu().numpy().reshape(self.input_shape).astype(self.input_dtype, copy=False)
        # Fallback path: PIL image (for testing/backward compatibility)
        elif isinstance(image, Image.Image):
            torch_input = self.preprocess(image.convert("RGB")).unsqueeze(0)  # type: ignore[operator,attr-defined]
            np_input = torch_input.cpu().numpy().reshape(self.input_shape).astype(self.input_dtype, copy=False)
        else:
            raise TypeError(f"Expected PIL Image or GPU tensor, got {type(image)}")
        
        # Copy to host buffer
        self.h_input[:] = np_input.ravel()  # type: ignore[index]
        
        # Transfer to device and execute
        cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)  # type: ignore[possibly-unbound,attr-defined]
        self.context.execute_async_v3(stream_handle=self.stream.handle)  # type: ignore[attr-defined]
        cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)  # type: ignore[possibly-unbound,attr-defined]
        self.stream.synchronize()  # type: ignore[attr-defined]
        
        # Convert output to torch tensor
        output = np.array(self.h_output).reshape(self.output_shape)  # type: ignore[arg-type]
        if output.dtype == np.float16:
            output = output.astype(np.float32)
        
        image_features = torch.from_numpy(output)
        
        # Normalize features (L2 normalization)
        image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def __del__(self):
        """Cleanup CUDA resources."""
        if self.d_input:
            self.d_input.free()
        if self.d_output:
            self.d_output.free()
