# models/smol_vlm_model.py
"""
SmolVLM model wrapper with ONNX runtime support.

This module provides the SmolVLMModel class which wraps the HuggingFace
SmolVLM model with optional ONNX runtime support for improved inference
performance. It handles model loading, input preparation, and text generation
with support for both standard transformers and ONNX execution.
"""

import torch
import onnxruntime
import numpy as np
import logging
from typing import List, Dict, Any, Generator
from transformers import AutoConfig
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText
#from transformers.models.idefics3.image_processing_idefics3 import Idefics3ImageProcessor
#from transformers import AutoModelForVision2Seq
from transformers.image_utils import load_image
from PIL import Image
from prompt.prompt import Prompt, History

from models.SmolVLM.model_config import ModelConfig

logger = logging.getLogger(__name__)

LOCAL_MODEL_PATH = "/home/patrick/smolvlm_local"

class SmolVLMModel:
    """
    SmolVLM model with ONNX runtime for efficient inference.

    This class provides a complete wrapper around the SmolVLM model with support
    for both standard HuggingFace transformers inference and optimized ONNX
    runtime execution. It handles model loading, tokenization, and generation.
    """

    def __init__(self, config: ModelConfig, use_onnx: bool = True):
        """
        Initialize the SmolVLM model with configuration options.

        Loads the model components including tokenizer, processor, and model weights.
        Optionally sets up ONNX runtime sessions for optimized inference.

        Args:
            config: Model configuration containing paths and parameters
            use_onnx: Whether to use ONNX runtime for faster inference

        Raises:
            Exception: If model loading fails or ONNX setup encounters errors
        """
        self._config = config
        self._use_onnx = use_onnx

        logger.info(f"Loading model from: {config.model_path}")

        # Load core transformers components
        self._load_transformers_components()

        # Initialize ONNX runtime if requested
        if use_onnx:
            self._load_onnx_sessions()
            self._setup_onnx_config()

        # Initialize metrics tracking (commented out - uncomment to enable)
        # from utils.vlm_metrics import create_vlm_metrics_tracker
        # self.metrics = create_vlm_metrics_tracker(self, enabled=True)
        self.metrics = None  # Placeholder - replace with line above to enable
    
    @property
    def config(self) -> ModelConfig:
        """Get the model configuration."""
        return self._config

    @property
    def use_onnx(self) -> bool:
        """Check if ONNX runtime is being used."""
        return self._use_onnx

    def _load_transformers_components(self):
        """
        Load the core transformers model components.

        Initializes the model, configuration, processor, tokenizer, and image
        processor from HuggingFace Hub or local path. Uses bfloat16 precision
        for memory efficiency.

        Raises:
            Exception: If any component fails to load
        """
        
#        AutoProcessor.register("Idefics3ImageProcessor", Idefics3ImageProcessor)
        
        self._model_config = AutoConfig.from_pretrained(self._config.model_path)
        self._processor = AutoProcessor.from_pretrained(self._config.model_path,)

        self._model = AutoModelForImageTextToText.from_pretrained(
            self._config.model_path,
            torch_dtype=torch.bfloat16,
            )

        self._image_processor = self._processor.image_processor
        self._tokenizer = self._processor.tokenizer

        # Initialize processor and model
        #self._model = AutoModelForVision2Seq.from_pretrained(
        #    "HuggingFaceTB/SmolVLM-256M-Instruct",
        #    torch_dtype=torch.bfloat16)

        # Analyze and log special token information
        #self._analyze_special_tokens()
    
    def _load_onnx_sessions(self):
        """
        Load ONNX runtime inference sessions for optimized execution.

        Attempts to load pre-exported ONNX models from the configured ONNX path.
        Falls back to transformers if ONNX models are not available or path doesn't exist.

        Raises:
            Exception: ONNX loading errors are caught and logged
        """
        try:
            from utils.onnx_utils import get_onnx_file_paths, setup_onnx_environment
            from config import get_config

            config = get_config()

            # Use ONNX utilities to check if ONNX can be used
            if not setup_onnx_environment(config):
                self._use_onnx = False
                return

            # Get ONNX model path and file paths
            onnx_model_path = config.model.get_onnx_model_path()
            onnx_files = get_onnx_file_paths(onnx_model_path)

            # Load ONNX sessions
            self._vision_session = onnxruntime.InferenceSession(str(onnx_files["vision_encoder"]))
            self._embed_session = onnxruntime.InferenceSession(str(onnx_files["embed_tokens"]))
            self._decoder_session = onnxruntime.InferenceSession(str(onnx_files["decoder"]))

            logger.info(f"ONNX sessions loaded successfully from: {onnx_model_path}")

        except Exception as e:
            logger.error(f"Failed to load ONNX sessions: {e}")
            logger.info("Falling back to transformers inference")
            self._use_onnx = False
    
    def _setup_onnx_config(self):
        """
        Setup configuration values required for ONNX inference.

        Extracts model architecture parameters needed for ONNX runtime
        execution, including attention head dimensions and special token IDs.
        """
        if not self._use_onnx:
            return

        # Extract text model configuration for ONNX setup
        text_config = self._model_config.text_config
        self._num_key_value_heads = text_config.num_key_value_heads
        self._head_dim = text_config.head_dim
        self._num_hidden_layers = text_config.num_hidden_layers
        self._eos_token_id = text_config.eos_token_id
        self._image_token_id = self._model_config.image_token_id
    
    def _analyze_special_tokens(self):
        """
        Analyze and log special token information for debugging.

        Examines special tokens used by the model to ensure they are properly
        configured in the tokenizer vocabulary. Logs token information for
        troubleshooting tokenization issues.
        """
        special_token = self._config.special_tokens["end_of_utterance"]
        is_in_vocab = special_token in self._tokenizer.vocab
        token_id = self._tokenizer.convert_tokens_to_ids(special_token)

        logger.info(f"Special token '{special_token}' in vocabulary: {is_in_vocab}")
        logger.info(f"Token ID for '{special_token}': {token_id}")
        logger.info(f"EOS token: {self._tokenizer.eos_token}")
        logger.info(f"EOS token ID: {self._tokenizer.eos_token_id}")
    
    def prepare_onnx_inputs(self, messages: List[Dict], images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """
        Prepare model inputs from messages and images.

        Converts chat messages and images into the format expected by the model,
        including tokenization, image preprocessing, and tensor preparation.

        Args:
            messages: Chat messages in the expected conversation format
            images: List of PIL images to process alongside text

        Returns:
            Dict[str, np.ndarray]: Dictionary containing model input tensors
        """
        # Apply chat template to format conversation messages
        prompt = self._processor.apply_chat_template(messages, add_generation_prompt=True)
        # Process text and images into model input format
        inputs = self._processor(text=prompt, images=images, return_tensors="np")
        return inputs
    
    def _initialize_past_key_values(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Initialize past key values for autoregressive generation.

        Creates empty key-value cache tensors for all transformer layers,
        used to maintain attention context during sequential token generation.

        Args:
            batch_size: Number of sequences in the batch

        Returns:
            Dict[str, np.ndarray]: Dictionary of zero-initialized key-value tensors
        """
        return {
            f'past_key_values.{layer}.{kv}': np.zeros(
                [batch_size, self._num_key_value_heads, 0, self._head_dim],
                dtype=np.float32
            )
            for layer in range(self._num_hidden_layers)
            for kv in ('key', 'value')
        }
    
    def generate_onnx(self, inputs: Dict[str, np.ndarray], max_new_tokens: int = None) -> Generator[str, None, None]:
        """
        Generate text using ONNX runtime with streaming support.

        Performs autoregressive text generation using ONNX runtime sessions
        for efficient inference. Yields decoded tokens as they are generated.

        Args:
            inputs: Prepared model inputs containing tokenized text and images
            max_new_tokens: Maximum number of new tokens to generate

        Yields:
            str: Decoded text tokens as they are generated

        Raises:
            RuntimeError: If ONNX runtime is not available or configured
        """
        if not self._use_onnx:
            raise RuntimeError("ONNX runtime not available")

        if max_new_tokens is None:
            max_new_tokens = self._config.max_new_tokens

        # Metrics tracking setup (commented out - uncomment to enable)
        # if hasattr(self, 'metrics'):
        #     input_tokens = inputs['input_ids'].shape[1] if 'input_ids' in inputs else 0
        #     input_images = inputs['pixel_values'].shape[0] if 'pixel_values' in inputs else 0
        #     self.metrics.record_input_processing(
        #         input_tokens=input_tokens,
        #         input_images=input_images,
        #         image_resolution=(224, 224)  # Adjust based on actual resolution
        #     )

        # Track total inference time (commented out - uncomment to enable)
        # if hasattr(self, 'metrics'):
        #     with self.metrics.track_total_inference() as tracker:
        #         yield from self._generate_onnx_with_metrics(inputs, max_new_tokens, tracker)
        # else:
        yield from self._generate_onnx_core(inputs, max_new_tokens)

    def _generate_onnx_core(self, inputs: Dict[str, np.ndarray], max_new_tokens: int) -> Generator[str, None, None]:
        """Core ONNX generation logic separated for metrics integration."""
        # Initialize generation state
        batch_size = inputs['input_ids'].shape[0]
        past_key_values = self._initialize_past_key_values(batch_size)
        image_features = None
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = np.cumsum(inputs['attention_mask'], axis=-1)

        generated_tokens = np.array([[]], dtype=np.int64)

        # Autoregressive generation loop
        for i in range(max_new_tokens):
            # Convert token IDs to embeddings
            # Metrics: Track token embedding (commented out - uncomment to enable)
            # if hasattr(self, 'metrics') and hasattr(tracker, 'track_token_embedding'):
            #     with tracker.track_token_embedding(token_count=input_ids.size):
            #         inputs_embeds = self._embed_session.run(None, {'input_ids': input_ids})[0]
            # else:
            inputs_embeds = self._embed_session.run(None, {'input_ids': input_ids})[0]

            # Process image features on first iteration
            if image_features is None:
                # Metrics: Track vision encoding (commented out - uncomment to enable)
                # if hasattr(self, 'metrics') and hasattr(tracker, 'track_vision_encoding'):
                #     input_images = inputs['pixel_values'].shape[0] if 'pixel_values' in inputs else 1
                #     with tracker.track_vision_encoding(image_count=input_images):
                #         image_features = self._vision_session.run(
                #             ['image_features'],
                #             {
                #                 'pixel_values': inputs['pixel_values'],
                #                 'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
                #             }
                #         )[0]
                # else:
                image_features = self._vision_session.run(
                    ['image_features'],
                    {
                        'pixel_values': inputs['pixel_values'],
                        'pixel_attention_mask': inputs['pixel_attention_mask'].astype(np.bool_)
                    }
                )[0]
                print(image_features.shape)
                print(inputs['pixel_values'].shape)
                print(inputs['pixel_attention_mask'].shape)

                # Replace image token embeddings with actual image features
                inputs_embeds[inputs['input_ids'] == self._image_token_id] = \
                    image_features.reshape(-1, image_features.shape[-1])

            # Run transformer decoder
            # Metrics: Track text generation step (commented out - uncomment to enable)
            # if hasattr(self, 'metrics') and hasattr(tracker, 'track_text_generation_step'):
            #     with tracker.track_text_generation_step(step_number=i+1):
            #         logits, *present_key_values = self._decoder_session.run(None, dict(
            #             inputs_embeds=inputs_embeds,
            #             attention_mask=attention_mask,
            #             position_ids=position_ids,
            #             **past_key_values,
            #         ))
            # else:
            logits, *present_key_values = self._decoder_session.run(None, dict(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                **past_key_values,
            ))

            # Select next token and update generation state
            input_ids = logits[:, -1].argmax(-1, keepdims=True)
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1

            # Update key-value cache for next iteration
            for j, key in enumerate(past_key_values):
                past_key_values[key] = present_key_values[j]

            generated_tokens = np.concatenate([generated_tokens, input_ids], axis=-1)

            # Metrics: Record generation step (commented out - uncomment to enable)
            # if hasattr(self, 'metrics') and hasattr(tracker, 'record_generation_step'):
            #     tracker.record_generation_step(
            #         token_id=int(input_ids[0, 0]),
            #         step_number=i+1
            #     )

            # Check for end-of-sequence token
            if (input_ids == self._config.eos_token_id).any():
                break

            # Yield decoded token for streaming
            yield self._processor.decode(input_ids[0])

        # Metrics: Export metrics after generation (commented out - uncomment to enable)
        # if hasattr(self, 'metrics'):
        #     try:
        #         metrics_file = self.metrics.export_metrics()
        #         logger.info(f"Metrics exported to: {metrics_file}")
        #     except Exception as e:
        #         logger.warning(f"Failed to export metrics: {e}")
                   
    def prepare_transformers_inputs(self, messages: List[Dict], images: List[Image.Image]) -> Dict[str, np.ndarray]:
        inputs = self._processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device, dtype=torch.bfloat16)
        return inputs


    def generate_transformers(self, inputs: Dict[str, Any], max_new_tokens: int = None) -> str:
        """
        Generate text using transformers library (fallback method).

        Uses the standard HuggingFace transformers generate method when ONNX
        runtime is not available or disabled. Provides the same functionality
        with potentially slower inference speed.

        Args:
            inputs: Prepared model inputs from processor
            max_new_tokens: Maximum number of new tokens to generate

        Returns:
            str: Complete generated text response
        """
        if max_new_tokens is None:
            max_new_tokens = self._config.max_new_tokens

        generated_ids = self._model.generate(**inputs, do_sample=False, max_new_tokens=64)
        generated_texts = self._processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        return generated_texts[0]
    
    def get_messages(self, prompt: Prompt) -> List[Dict[str, Any]]:
        """
        Convert a Prompt object into the message format expected by the model.

        Formats the conversation history and current user input into the structured
        message format required by the chat template processor.

        Args:
            prompt: Prompt object containing conversation history and current input

        Returns:
            List[Dict[str, Any]]: List of formatted message dictionaries
        """
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
                    {"type": "image", "image": ""},
                ]
            },
        ]
        return messages
