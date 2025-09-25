# models/smol_vlm_model.py
"""SmolVLM model wrapper with ONNX runtime support."""

import torch
import onnxruntime
import numpy as np
import logging
from typing import List, Dict, Any, Generator
from transformers import AutoConfig, AutoProcessor, AutoModelForImageTextToText
from PIL import Image
from src.prompt.prompt import Prompt, History

from src.models.SmolVLM.model_config import ModelConfig

logger = logging.getLogger(__name__)

class SmolVLMModel:
    """SmolVLM model with ONNX runtime for efficient inference."""
    
    def __init__(self, config: ModelConfig, use_onnx: bool = True):
        """
        Initialize the SmolVLM model.
        
        Args:
            config: Model configuration
            use_onnx: Whether to use ONNX runtime for inference
        """
        self.config = config
        self.use_onnx = use_onnx
        
        logger.info(f"Loading model from: {config.model_path}")
        
        # Load transformers components
        self._load_transformers_components()
        
        # Load ONNX sessions if specified
        if use_onnx:
            self._load_onnx_sessions()
            self._setup_onnx_config()
    
    def _load_transformers_components(self):
        """Load the transformers model components."""
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.config.model_path, 
            torch_dtype=torch.bfloat16
        )
        self.model_config = AutoConfig.from_pretrained(self.config.model_path)
        self.processor = AutoProcessor.from_pretrained(self.config.model_path)
        self.image_processor = self.processor.image_processor
        self.tokenizer = self.processor.tokenizer
        
        # Analyze special tokens
        self._analyze_special_tokens()
    
    def _load_onnx_sessions(self):
        """Load ONNX runtime sessions."""
        try:
            self.vision_session = onnxruntime.InferenceSession("vision_encoder.onnx")
            self.embed_session = onnxruntime.InferenceSession("embed_tokens.onnx")
            self.decoder_session = onnxruntime.InferenceSession("decoder_model_merged.onnx")
            logger.info("ONNX sessions loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load ONNX sessions: {e}")
            self.use_onnx = False
    
    def _setup_onnx_config(self):
        """Setup configuration values for ONNX inference."""
        if not self.use_onnx:
            return
            
        text_config = self.model_config.text_config
        self.num_key_value_heads = text_config.num_key_value_heads
        self.head_dim = text_config.head_dim
        self.num_hidden_layers = text_config.num_hidden_layers
        self.eos_token_id = text_config.eos_token_id
        self.image_token_id = self.model_config.image_token_id
    
    def _analyze_special_tokens(self):
        """Analyze and log special token information."""
        special_token = self.config.special_tokens["end_of_utterance"]
        is_in_vocab = special_token in self.tokenizer.vocab
        token_id = self.tokenizer.convert_tokens_to_ids(special_token)
        
        logger.info(f"Special token '{special_token}' in vocabulary: {is_in_vocab}")
        logger.info(f"Token ID for '{special_token}': {token_id}")
        logger.info(f"EOS token: {self.tokenizer.eos_token}")
        logger.info(f"EOS token ID: {self.tokenizer.eos_token_id}")
    
    def prepare_inputs(self, messages: List[Dict], images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for the model.
        
        Args:
            messages: Chat messages in the expected format
            images: List of PIL images
            
        Returns:
            Dictionary of prepared inputs
        """
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=images, return_tensors="np")
        return inputs
    
    def _initialize_past_key_values(self, batch_size: int) -> Dict[str, np.ndarray]:
        """Initialize past key values for generation."""
        return {
            f'past_key_values.{layer}.{kv}': np.zeros(
                [batch_size, self.num_key_value_heads, 0, self.head_dim], 
                dtype=np.float32
            )
            for layer in range(self.num_hidden_layers)
            for kv in ('key', 'value')
        }
    
    def generate_onnx(self, inputs: Dict[str, np.ndarray], max_new_tokens: int = None) -> Generator[np.ndarray, None, None]:
        """
        Generate text using ONNX runtime.
        
        Args:
            inputs: Prepared model inputs
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated token IDs
        """
        if not self.use_onnx:
            raise RuntimeError("ONNX runtime not available")
        
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        
        batch_size = inputs['input_ids'].shape[0]
        past_key_values = self._initialize_past_key_values(batch_size)
        image_features = None
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = np.cumsum(inputs['attention_mask'], axis=-1)
        
        generated_tokens = np.array([[]], dtype=np.int64)
        
        for i in range(max_new_tokens):
            # Get input embeddings
            inputs_embeds = self.embed_session.run(None, {'input_ids': input_ids})[0]
            
            # Compute vision features if not already computed
            if image_features is None:
                image_features = self.vision_session.run(
                    ['image_features'],
                    {
                        'pixel_values': inputs['pixel_values'],
                        'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
                    }
                )[0]
                
                # Merge text and vision embeddings
                inputs_embeds[inputs['input_ids'] == self.image_token_id] = \
                    image_features.reshape(-1, image_features.shape[-1])
            
            # Run decoder
            logits, *present_key_values = self.decoder_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **past_key_values,
            ))
            
            # Update for next iteration
            input_ids = logits[:, -1].argmax(-1, keepdims=True)
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1
            
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]
            
            generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)
            
            # Check for end of sequence
            if (input_ids == self.config.eos_token_id).any():
                break
            
            # Optional streaming output
            yield self.processor.decode(input_ids[0])
                   
    def generate_transformers(self, inputs: Dict[str, Any], max_new_tokens: int = None) -> str:
        """
        Generate text using transformers library (fallback method).
        
        Args:
            inputs: Prepared model inputs
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text
        """
        if max_new_tokens is None:
            max_new_tokens = self.config.max_new_tokens
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )
        
        return self.processor.batch_decode(outputs, skip_special_tokens=True)[0]

    def get_messages(self, prompt: Prompt) -> List[Dict[str, Any]]:
        """Get the current prompt from a Prompt object."""
        messages = [
            {
                "role": "system",
                "content": [
                {"type": "text", "text": "You are a helpful assistant."},
                {"type": "text", "text": prompt.history.get_formatted_history()},
                ]
            },
            {
                "role": "user",
                "content": [
                {"type": "text", "text": prompt.user_input},
                {"type": "image", "image": ""},                ]
            },
]
        # Assuming the Prompt class has a method to get the formatted prompt string
        return messages