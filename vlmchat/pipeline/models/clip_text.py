"""
CLIP text encoder for the pipeline.

This module provides OpenCLIP-based text encoding for semantic similarity.
"""

import logging
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

# Third-party imports
try:
    import open_clip
    from mobileclip.modules.common.mobileone import reparameterize_model  # type: ignore
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.warning("OpenCLIP not available - install open_clip_torch and mobileclip")

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("HuggingFace Hub not available - will not auto-download models")

try:
    import tensorrt as trt  # type: ignore
    import pycuda.autoinit  # type: ignore
    import pycuda.driver as cuda  # type: ignore
    import numpy as np
    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logger.warning("TensorRT not available")


class ClipTextOpenClip:
    """
    OpenCLIP text encoder for semantic similarity queries.
    
    Similar to YoloUltralytics pattern - simple, focused interface.
    Supports MobileCLIP models optimized for edge devices.
    """
    
    def __init__(self, 
                 model_name: str = "MobileCLIP2-S0",
                 model_path: Optional[str] = None,
                 device: str = "cpu"):
        """
        Initialize OpenCLIP text encoder.
        
        Args:
            model_name: Model name (e.g., "MobileCLIP2-S0", "ViT-B-32")
            model_path: Path to pretrained weights (optional, will download from HF)
            device: Device to run on ("cpu" or "cuda")
        """
        self.model_name = model_name
        self.device = device
        self._model = None
        self._tokenizer = None
        self._cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        
        if not OPEN_CLIP_AVAILABLE:
            raise RuntimeError("OpenCLIP not available - install open_clip_torch")
        
        # Resolve model path
        resolved_path = self._resolve_model_path(model_name, model_path)
        
        # Load model
        logger.info(f"Loading CLIP text model: {model_name} from {resolved_path}")
        
        # MobileCLIP-specific normalization (for consistency, though text encoder doesn't use)
        model_kwargs = {}
        if "MobileCLIP" in model_name:
            if not (model_name.endswith("S3") or model_name.endswith("S4") or model_name.endswith("L-14")):
                model_kwargs["image_mean"] = (0, 0, 0)
                model_kwargs["image_std"] = (1, 1, 1)
        
        try:
            self._model, _, _ = open_clip.create_model_and_transforms(
                model_name,
                pretrained=resolved_path,
                **model_kwargs
            )
            self._tokenizer = open_clip.get_tokenizer(model_name)
            
            self._model.eval()
            self._model = self._model.to(device)
            
            # Reparameterize MobileCLIP for inference optimization
            if "MobileCLIP" in model_name:
                logger.info("Reparameterizing MobileCLIP text model...")
                self._model = reparameterize_model(self._model)
            
            logger.info(f"CLIP text model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load CLIP text model: {e}")
            raise
    
    def _resolve_model_path(self, model_name: str, model_path: Optional[str]) -> str:
        """
        Resolve model path: use provided path, or download from HuggingFace.
        
        Args:
            model_name: Model name
            model_path: Optional explicit path
            
        Returns:
            Resolved path to model weights
        """
        # If explicit path provided and exists, use it
        if model_path:
            path = Path(model_path)
            if path.exists():
                return str(path)
            logger.warning(f"Provided path doesn't exist: {model_path}")
        
        # Try to download from HuggingFace
        if HF_HUB_AVAILABLE and "MobileCLIP" in model_name:
            try:
                repo_map = {
                    "MobileCLIP-S0": "apple/MobileCLIP-S0",
                    "MobileCLIP-S1": "apple/MobileCLIP-S1",
                    "MobileCLIP-S2": "apple/MobileCLIP-S2",
                    "MobileCLIP2-S0": "apple/MobileCLIP2-S0",
                    "MobileCLIP2-S1": "apple/MobileCLIP2-S1",
                    "MobileCLIP2-S2": "apple/MobileCLIP2-S2",
                    "MobileCLIP2-S3": "apple/MobileCLIP2-S3",
                    "MobileCLIP2-S4": "apple/MobileCLIP2-S4",
                    "MobileCLIP2-B": "apple/MobileCLIP2-B",
                    "MobileCLIP2-L-14": "apple/MobileCLIP2-L-14",
                }
                
                repo_id = repo_map.get(model_name)
                if repo_id:
                    filename = f"{model_name.lower().replace('-', '_')}.pt"
                    logger.info(f"Downloading from HuggingFace: {repo_id}/{filename}")
                    downloaded_path = hf_hub_download(
                        repo_id=repo_id,
                        filename=filename,
                        cache_dir=None
                    )
                    logger.info(f"Downloaded to: {downloaded_path}")
                    return downloaded_path
                    
            except Exception as e:
                logger.warning(f"HuggingFace download failed: {e}")
        
        # Fallback to default/pretrained
        fallback = model_path or "openai"  # OpenCLIP will try openai weights
        logger.info(f"Using fallback: {fallback}")
        return fallback
    
    def encode(self, texts: List[str], use_cache: bool = True) -> torch.Tensor:
        """
        Encode text prompts into normalized feature vectors.
        
        Args:
            texts: List of text strings to encode
            use_cache: Whether to use cached encodings
            
        Returns:
            Normalized feature tensor of shape (len(texts), feature_dim)
        """
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        # Check cache
        if use_cache:
            cache_key = tuple(texts)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Tokenize
        text_tokens = self._tokenizer(texts)
        text_tokens = text_tokens.to(self.device)
        
        # Encode
        with torch.no_grad():
            if self.device == "cuda" and torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    features = self._model.encode_text(text_tokens)
            else:
                features = self._model.encode_text(text_tokens)
            
            # Normalize
            features = features / features.norm(dim=-1, keepdim=True)
        
        # Cache result
        if use_cache:
            cache_key = tuple(texts)
            self._cache[cache_key] = features
        
        return features
    
    def clear_cache(self) -> None:
        """Clear cached encodings."""
        self._cache.clear()
    
    @property
    def feature_dim(self) -> int:
        """Get feature dimension of the model."""
        if self._model is None:
            return 512  # Default
        # Try to get from model
        try:
            return self._model.text_projection.shape[1]
        except:
            return 512
    
    def __repr__(self) -> str:
        return f"ClipTextOpenClip(model={self.model_name}, device={self.device})"


class ClipTextTensorRT:
    """
    TensorRT CLIP text encoder for GPU-accelerated inference.
    
    Requires a pre-built TensorRT engine file.
    """
    
    def __init__(self, engine_path: str, model_name: str = "MobileCLIP2-S0"):
        """
        Initialize TensorRT text encoder.
        
        Args:
            engine_path: Path to TensorRT engine file (.trt or .engine)
            model_name: Model name for tokenizer (e.g., "MobileCLIP2-S0")
        """
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available - install tensorrt and pycuda")
        
        if not OPEN_CLIP_AVAILABLE:
            raise RuntimeError("OpenCLIP not available - needed for tokenizer")
        
        self.engine_path = Path(engine_path)
        self.model_name = model_name
        self._cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        
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
        self.sequence_length = 77
        
        # Load engine
        if not self.engine_path.exists():
            raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
        
        logger.info(f"Loading TensorRT text engine: {self.engine_path}")
        
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
        input_name = "text_input"
        output_name = "text_features"
        
        self.input_shape = tuple(self.context.get_tensor_shape(input_name))
        self.output_shape = tuple(self.context.get_tensor_shape(output_name))
        self.input_dtype = trt.nptype(self.engine.get_tensor_dtype(input_name))
        self.output_dtype = trt.nptype(self.engine.get_tensor_dtype(output_name))
        
        # Extract batch size and sequence length
        self.engine_batch_size = self.input_shape[0] if self.input_shape else 1
        self.sequence_length = self.input_shape[1] if len(self.input_shape) > 1 else 77
        
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
        
        # Load tokenizer
        self._tokenizer = open_clip.get_tokenizer(model_name)
        
        logger.info("TensorRT text encoder loaded successfully")
    
    def encode(self, texts: List[str], use_cache: bool = True) -> torch.Tensor:
        """
        Encode text prompts using TensorRT.
        
        Args:
            texts: List of text strings to encode
            use_cache: Whether to use cached encodings
            
        Returns:
            Normalized feature tensor of shape (len(texts), feature_dim)
        """
        # Check cache
        if use_cache:
            cache_key = tuple(texts)
            if cache_key in self._cache:
                return self._cache[cache_key]
        
        # Tokenize
        tokenized = self._tokenizer(texts)
        
        # Convert to numpy
        if isinstance(tokenized, torch.Tensor):
            np_input = tokenized.cpu().numpy()
        else:
            np_input = np.array(tokenized)
        
        # Ensure correct shape
        batch_size = len(texts)
        if np_input.ndim == 1:
            np_input = np_input.reshape(batch_size, self.sequence_length)
        elif np_input.shape[0] != batch_size:
            np_input = np_input.reshape(batch_size, self.sequence_length)
        
        # Process in batches
        all_features = []
        for batch_start in range(0, batch_size, self.engine_batch_size):
            batch_end = min(batch_start + self.engine_batch_size, batch_size)
            batch_inputs = np_input[batch_start:batch_end]
            
            # Pad if needed
            actual_batch_size = batch_inputs.shape[0]
            if actual_batch_size < self.engine_batch_size:
                padding = np.zeros(
                    (self.engine_batch_size - actual_batch_size, self.sequence_length),
                    dtype=self.input_dtype
                )
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
        
        # Cache
        if use_cache:
            cache_key = tuple(texts)
            self._cache[cache_key] = features
        
        return features
    
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
        return f"ClipTextTensorRT(engine={self.engine_path.name}, model={self.model_name})"
