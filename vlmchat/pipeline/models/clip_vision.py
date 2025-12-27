"""
CLIP vision encoder for the pipeline.

This module provides TensorRT-based vision encoding for detection crop similarity.
"""

import logging
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Third-party imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available")

try:
    import tensorrt as trt  # type: ignore
    import pycuda.autoinit  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT not available")


class ClipVisionTensorRT:
    """
    TensorRT CLIP vision encoder for GPU-accelerated image embedding inference.
    
    Requires a pre-built TensorRT engine file for CLIP vision model.
    Follows the same pattern as ClipTextTensorRT for consistency.
    """
    
    def __init__(self, 
                 engine_path: str, 
                 model_name: str = "MobileCLIP2-S0",
                 input_size: int = 256,
                 mean: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                 std: Tuple[float, float, float] = (1.0, 1.0, 1.0)):
        """
        Initialize TensorRT vision encoder.
        
        Args:
            engine_path: Path to TensorRT engine file (.trt or .engine)
            model_name: Model name for reference (e.g., "MobileCLIP2-S0")
            input_size: Expected input image size (width/height, assumes square)
            mean: RGB mean for normalization
            std: RGB std for normalization
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available - install tensorrt and pycuda")
        
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch not available - needed for tensor operations")
        
        self.engine_path = Path(engine_path)
        self.model_name = model_name
        self.input_size = input_size
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)
        self._cache: Dict[int, torch.Tensor] = {}  # Cache by hash of image arrays
        
        # TensorRT components
        self.engine = None
        self.context = None
        self.stream = None
        
        # Buffers
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
        
        # Load engine
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        logger.info(f"Loading TensorRT vision engine: {self.engine_path}")
        
        # Load TensorRT engine
        trt_logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(trt_logger)
        
        with open(self.engine_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine")
        
        # Create execution context
        self.context = self.engine.create_execution_context()
        
        # Get tensor info (TensorRT 10+ API)
        input_name = "image_input"
        output_name = "image_features"
        
        # Get shapes (may have dynamic dimensions)
        self.input_shape = tuple(self.engine.get_tensor_shape(input_name))
        self.output_shape = tuple(self.engine.get_tensor_shape(output_name))
        
        # Handle dynamic batch size
        if self.input_shape[0] == -1:
            # Set to batch size 1 for execution
            self.engine_batch_size = 1
            self.input_shape = (1,) + self.input_shape[1:]
            self.output_shape = (1,) + self.output_shape[1:]
            self.context.set_input_shape(input_name, self.input_shape)
        else:
            self.engine_batch_size = self.input_shape[0]
        
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(output_name))
        
        # Expected: (batch, channels, height, width)
        if len(self.input_shape) >= 4:
            self.channels = self.input_shape[1]
            self.height = self.input_shape[2]
            self.width = self.input_shape[3]
        else:
            self.channels = 3
            self.height = self.input_size
            self.width = self.input_size
        
        logger.info(f"  Input: {self.input_shape}, dtype: {self.input_dtype}")
        logger.info(f"  Output: {self.output_shape}, dtype: {self.output_dtype}")
        logger.info(f"  Batch size: {self.engine_batch_size}")
        
        # Allocate buffers
        input_size = int(np.prod(self.input_shape))
        output_size = int(np.prod(self.output_shape))
        
        self.h_input = cuda.pagelocked_empty(input_size, self.input_dtype)
        self.h_output = cuda.pagelocked_empty(output_size, self.output_dtype)
        
        self.d_input = cuda.mem_alloc(self.h_input.nbytes)
        self.d_output = cuda.mem_alloc(self.h_output.nbytes)
        
        # Bind tensors
        self.context.set_tensor_address(input_name, int(self.d_input))
        self.context.set_tensor_address(output_name, int(self.d_output))
        
        # Create CUDA stream
        self.stream = cuda.Stream()
        
        logger.info("TensorRT vision encoder loaded successfully")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for CLIP vision model.
        
        Args:
            image: BGR image array (H, W, 3) from OpenCV
            
        Returns:
            Preprocessed array (C, H, W) in RGB, normalized
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize to input size
        if image_rgb.shape[:2] != (self.height, self.width):
            image_rgb = cv2.resize(image_rgb, (self.width, self.height), 
                                   interpolation=cv2.INTER_LINEAR)
        
        # Convert to float32 and normalize to [0, 1]
        image_float = image_rgb.astype(np.float32) / 255.0
        
        # Apply mean/std normalization
        image_normalized = (image_float - self.mean) / self.std
        
        # Convert HWC to CHW
        image_chw = np.transpose(image_normalized, (2, 0, 1))
        
        return image_chw
    
    def encode(self, images: List[np.ndarray], use_cache: bool = True) -> torch.Tensor:
        """
        Encode image crops using TensorRT.
        
        Args:
            images: List of BGR image arrays (H, W, 3) from OpenCV
            use_cache: Whether to use cached encodings (based on array hash)
            
        Returns:
            Normalized feature tensor of shape (len(images), feature_dim)
        """
        # Preprocess all images
        preprocessed = []
        cache_hits = []
        
        for img in images:
            # Check cache
            if use_cache:
                img_hash = hash(img.tobytes())
                if img_hash in self._cache:
                    cache_hits.append(self._cache[img_hash])
                    preprocessed.append(None)  # Placeholder
                    continue
            
            # Preprocess
            processed = self._preprocess_image(img)
            preprocessed.append(processed)
        
        # If all cache hits, return stacked results
        if all(p is None for p in preprocessed):
            return torch.stack(cache_hits)
        
        # Filter out cache hits for batch processing
        batch_images = [p for p in preprocessed if p is not None]
        if not batch_images:
            return torch.stack(cache_hits)
        
        batch_size = len(batch_images)
        
        # Stack into batch array
        batch_array = np.stack(batch_images, axis=0)  # (N, C, H, W)
        
        # Process in engine batches
        all_features = []
        for batch_start in range(0, batch_size, self.engine_batch_size):
            batch_end = min(batch_start + self.engine_batch_size, batch_size)
            batch_inputs = batch_array[batch_start:batch_end]
            
            # Pad if needed
            actual_batch_size = batch_inputs.shape[0]
            if actual_batch_size < self.engine_batch_size:
                padding_shape = (self.engine_batch_size - actual_batch_size,) + batch_inputs.shape[1:]
                padding = np.zeros(padding_shape, dtype=self.input_dtype)
                batch_inputs = np.vstack([batch_inputs, padding])
            
            # Convert dtype
            batch_inputs = batch_inputs.astype(self.input_dtype, copy=False)
            
            # Copy to host buffer
            self.h_input[:] = batch_inputs.ravel()
            
            # Execute
            cuda.memcpy_htod_async(self.d_input, self.h_input, self.stream)
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.h_output, self.d_output, self.stream)
            self.stream.synchronize()
            
            # Get output
            output = np.array(self.h_output).reshape(self.output_shape)
            if output.dtype == np.float16:
                output = output.astype(np.float32)
            
            # Remove padding
            output = output[:actual_batch_size]
            all_features.append(output)
        
        # Convert to torch and normalize
        features = torch.from_numpy(np.vstack(all_features))
        features = features / features.norm(dim=-1, keepdim=True)
        
        # Cache new embeddings
        if use_cache:
            feature_idx = 0
            for i, (img, processed) in enumerate(zip(images, preprocessed)):
                if processed is not None:  # Not a cache hit
                    img_hash = hash(img.tobytes())
                    self._cache[img_hash] = features[feature_idx]
                    feature_idx += 1
        
        # Merge cache hits with new features
        result = []
        cache_idx = 0
        feature_idx = 0
        for processed in preprocessed:
            if processed is None:
                result.append(cache_hits[cache_idx])
                cache_idx += 1
            else:
                result.append(features[feature_idx])
                feature_idx += 1
        
        return torch.stack(result)
    
    def clear_cache(self) -> None:
        """Clear cached encodings."""
        self._cache.clear()
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension."""
        if self.output_shape:
            return self.output_shape[-1]
        return 512
    
    def __del__(self):
        """Cleanup CUDA resources."""
        if hasattr(self, 'd_input') and self.d_input:
            self.d_input.free()
        if hasattr(self, 'd_output') and self.d_output:
            self.d_output.free()
    
    def __repr__(self) -> str:
        return f"ClipVisionTensorRT(engine={self.engine_path.name}, model={self.model_name})"
