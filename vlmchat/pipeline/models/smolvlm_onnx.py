"""
High-level SmolVLM ONNX wrapper coordinating vision, embedding, and decoder backends.

Provides a simple interface for multimodal text generation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Any, Generator, Optional

import numpy as np
from PIL import Image

from .smolvlm_vision_onnx import SmolVLMVisionOnnx
from .smolvlm_embed_onnx import SmolVLMEmbedOnnx
from .smolvlm_decoder_onnx import SmolVLMDecoderOnnx

logger = logging.getLogger(__name__)

# Check for transformers (for processor/tokenizer)
try:
    from transformers import AutoProcessor, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False


class SmolVLMOnnx:
    """
    High-level SmolVLM ONNX wrapper.
    
    Coordinates vision encoder, token embeddings, and decoder for
    multimodal text generation.
    """
    
    def __init__(self,
                 model_path: str,
                 vision_engine: Optional[str] = None,
                 embed_engine: Optional[str] = None,
                 decoder_engine: Optional[str] = None,
                 device: str = "cuda"):
        """
        Initialize SmolVLM with ONNX backends.
        
        Args:
            model_path: Path to model directory (for processor/config) or ONNX files
            vision_engine: Path to vision_encoder.onnx (default: model_path/vision_encoder.onnx)
            embed_engine: Path to embed_tokens.onnx (default: model_path/embed_tokens.onnx)
            decoder_engine: Path to decoder_model_merged.onnx (default: model_path/decoder_model_merged.onnx)
            device: Device for inference ('cuda' or 'cpu')
        """
        self.model_path = Path(model_path)
        self.device = device
        
        # Resolve engine paths
        if vision_engine is None:
            vision_engine = str(self.model_path / "vision_encoder.onnx")
        if embed_engine is None:
            embed_engine = str(self.model_path / "embed_tokens.onnx")
        if decoder_engine is None:
            decoder_engine = str(self.model_path / "decoder_model_merged.onnx")
        
        logger.info(f"Initializing SmolVLM ONNX wrapper")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Device: {device}")
        
        # Load processor and config (for tokenization and model parameters)
        self.processor = None
        self.tokenizer = None
        self.config = None
        self.image_token_id = None
        self.eos_token_id = []
        
        if TRANSFORMERS_AVAILABLE:
            try:
                # Try to load from HuggingFace model
                self.config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
                self.processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-256M-Instruct")
                self.tokenizer = self.processor.tokenizer
                
                # Extract model parameters
                text_config = self.config.text_config
                self.num_hidden_layers = text_config.num_hidden_layers
                self.num_key_value_heads = text_config.num_key_value_heads
                self.head_dim = text_config.head_dim
                self.image_token_id = self.config.image_token_id
                
                # Handle EOS token IDs
                cfg_eos = text_config.eos_token_id
                if cfg_eos is not None:
                    if isinstance(cfg_eos, (list, tuple)):
                        self.eos_token_id = list(cfg_eos)
                    else:
                        self.eos_token_id = [int(cfg_eos)]
                
                # Add tokenizer EOS if different
                if self.tokenizer.eos_token_id not in self.eos_token_id:
                    self.eos_token_id.append(self.tokenizer.eos_token_id)
                
                logger.info(f"  Loaded processor and config")
                logger.info(f"  Model: {self.num_hidden_layers} layers, {self.num_key_value_heads} KV heads")
                logger.info(f"  Image token ID: {self.image_token_id}")
                logger.info(f"  EOS token IDs: {self.eos_token_id}")
                
            except Exception as e:
                logger.warning(f"Failed to load processor/config: {e}")
                logger.warning("Will require externally prepared inputs")
                # Use default parameters for SmolVLM-256M
                self.num_hidden_layers = 32
                self.num_key_value_heads = 8
                self.head_dim = 64
        else:
            logger.warning("Transformers not available - will require externally prepared inputs")
            # Use default parameters for SmolVLM-256M
            self.num_hidden_layers = 32
            self.num_key_value_heads = 8
            self.head_dim = 64
        
        # Initialize backends
        logger.info("Loading ONNX backends...")
        self.vision = SmolVLMVisionOnnx(vision_engine, device=device)
        self.embed = SmolVLMEmbedOnnx(embed_engine, device=device)
        self.decoder = SmolVLMDecoderOnnx(
            decoder_engine, 
            num_hidden_layers=self.num_hidden_layers,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            device="cpu"  # Decoder runs on CPU
        )
        
        logger.info("SmolVLM ONNX wrapper initialized successfully")
    
    def prepare_inputs(self, 
                      messages: List[Dict[str, str]], 
                      images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for generation.
        
        Args:
            messages: Chat messages in format [{"role": "user", "content": "..."}, ...]
            images: List of PIL images
            
        Returns:
            Dictionary with input_ids, pixel_values, pixel_attention_mask, attention_mask
        """
        if self.processor is None:
            raise RuntimeError("Processor not available - provide prepared inputs directly")
        
        # Apply chat template
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process inputs
        inputs = self.processor(text=prompt, images=images, return_tensors="np")
        
        return inputs
    
    def generate(self,
                inputs: Dict[str, np.ndarray],
                max_new_tokens: int = 100) -> str:
        """
        Generate text from inputs.
        
        Args:
            inputs: Prepared inputs (input_ids, pixel_values, etc.)
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text string
        """
        tokens = []
        for token_id in self.generate_stream(inputs, max_new_tokens):
            tokens.append(token_id)
        
        if self.tokenizer:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        else:
            return f"Generated {len(tokens)} tokens (no tokenizer available)"
    
    def generate_stream(self,
                       inputs: Dict[str, np.ndarray],
                       max_new_tokens: int = 100) -> Generator[int, None, None]:
        """
        Stream generation token by token.
        
        Args:
            inputs: Prepared inputs (input_ids, pixel_values, etc.)
            max_new_tokens: Maximum tokens to generate
            
        Yields:
            Generated token IDs
        """
        batch_size = inputs['input_ids'].shape[0]
        
        # Initialize past key-values
        past_key_values = self.decoder.initialize_past_key_values(batch_size)
        
        # Encode vision features (once, lazily)
        image_features = None
        
        # Prepare initial inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = np.cumsum(attention_mask, axis=-1)
        
        # Generation loop
        for i in range(max_new_tokens):
            # Get token embeddings
            inputs_embeds = self.embed.embed(input_ids)
            
            # Inject image features on first step using ORIGINAL input_ids positions
            if image_features is None:
                image_features = self.vision.encode(
                    inputs['pixel_values'],
                    inputs['pixel_attention_mask']
                )
                
                if self.image_token_id is not None:
                    # Replace image token embeddings with vision features using ORIGINAL prompt
                    image_mask = inputs['input_ids'] == self.image_token_id
                    if image_mask.any():
                        inputs_embeds[image_mask] = image_features.reshape(-1, image_features.shape[-1])
            
            # Decoder step
            logits, past_key_values = self.decoder.decode(
                inputs_embeds,
                attention_mask,
                position_ids,
                past_key_values
            )
            
            # Sample next token (greedy)
            next_token_id = int(logits[:, -1].argmax(-1)[0])
            
            # Check for EOS
            if next_token_id in self.eos_token_id:
                break
            
            # Yield token
            yield next_token_id
            
            # Update for next step
            input_ids = np.array([[next_token_id]], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1
    
    def __repr__(self) -> str:
        return (f"SmolVLMOnnx(model={self.model_path.name}, device={self.device}, "
                f"layers={self.num_hidden_layers})")
