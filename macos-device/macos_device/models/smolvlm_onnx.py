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

try:
    from transformers import AutoProcessor, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Default architecture parameters per model size
_ARCH_DEFAULTS = {
    "256M": {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 64},
    "500M": {"num_hidden_layers": 32, "num_key_value_heads": 8, "head_dim": 64},
}


class SmolVLMOnnx:
    """
    High-level SmolVLM ONNX wrapper.

    Coordinates vision encoder, token embeddings, and decoder for
    multimodal text generation.
    """

    def __init__(self,
                 model_path: str,
                 model_size: str = "256M",
                 vision_engine: Optional[str] = None,
                 embed_engine: Optional[str] = None,
                 decoder_engine: Optional[str] = None,
                 device: str = "cuda"):
        """
        Initialize SmolVLM with ONNX backends.

        Args:
            model_path: Path to model directory (for processor/config and default ONNX paths)
            model_size: HuggingFace model size suffix, e.g. "256M" or "500M"
            vision_engine: Override path to vision_encoder.onnx
            embed_engine: Override path to embed_tokens.onnx
            decoder_engine: Override path to decoder_model_merged.onnx
            device: Device for vision/embed inference ('cuda' or 'cpu')
        """
        self.model_path = Path(model_path)
        self.model_size = model_size
        self.device = device

        # Resolve engine paths
        vision_engine = vision_engine or str(self.model_path / "vision_encoder.onnx")
        embed_engine = embed_engine or str(self.model_path / "embed_tokens.onnx")
        decoder_engine = decoder_engine or str(self.model_path / "decoder_model_merged.onnx")

        logger.info(f"Initializing SmolVLM ONNX wrapper ({model_size})")
        logger.info(f"  Model path: {model_path}")
        logger.info(f"  Device: {device}")

        # Processor / tokenizer / config (from HuggingFace hub or local)
        self.processor = None
        self.tokenizer = None
        self.config = None
        self.image_token_id = None
        self.eos_token_id: List[int] = []

        # Architecture defaults (overwritten if transformers loads config)
        defaults = _ARCH_DEFAULTS.get(model_size, _ARCH_DEFAULTS["256M"])
        self.num_hidden_layers = defaults["num_hidden_layers"]
        self.num_key_value_heads = defaults["num_key_value_heads"]
        self.head_dim = defaults["head_dim"]

        if TRANSFORMERS_AVAILABLE:
            hf_name = f"HuggingFaceTB/SmolVLM2-{model_size}-Instruct"
            try:
                self.config = AutoConfig.from_pretrained(hf_name)
                self.processor = AutoProcessor.from_pretrained(hf_name)
                self.tokenizer = self.processor.tokenizer

                text_config = self.config.text_config
                self.num_hidden_layers = text_config.num_hidden_layers
                self.num_key_value_heads = text_config.num_key_value_heads
                self.head_dim = text_config.head_dim
                self.image_token_id = self.config.image_token_id

                cfg_eos = text_config.eos_token_id
                if cfg_eos is not None:
                    self.eos_token_id = list(cfg_eos) if isinstance(cfg_eos, (list, tuple)) else [int(cfg_eos)]
                if self.tokenizer.eos_token_id not in self.eos_token_id:
                    self.eos_token_id.append(self.tokenizer.eos_token_id)

                logger.info(f"  Loaded config: {self.num_hidden_layers} layers, {self.num_key_value_heads} KV heads")
                logger.info(f"  Image token ID: {self.image_token_id} | EOS: {self.eos_token_id}")
            except Exception as e:
                logger.warning(f"Failed to load processor/config from {hf_name}: {e}")
                logger.warning("Using default architecture parameters – provide prepared inputs directly")
        else:
            logger.warning("transformers not available – provide prepared inputs directly")

        # Initialize backends
        logger.info("Loading ONNX backends...")
        self.vision = SmolVLMVisionOnnx(vision_engine, device=device)
        self.embed = SmolVLMEmbedOnnx(embed_engine, device=device)
        self.decoder = SmolVLMDecoderOnnx(
            decoder_engine,
            num_hidden_layers=self.num_hidden_layers,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            device="cpu",   # decoder runs on CPU for both 256M and 500M
            model_size=model_size,
        )
        logger.info("SmolVLM ONNX wrapper initialized successfully")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prepare_inputs(self,
                       messages: List[Dict[str, Any]],
                       images: List[Image.Image]) -> Dict[str, np.ndarray]:
        """
        Prepare inputs for generation using the HuggingFace processor.

        Args:
            messages: Chat messages, e.g. [{"role": "user", "content": "..."}]
            images: List of PIL images

        Returns:
            Dict with input_ids, pixel_values, pixel_attention_mask, attention_mask
        """
        if self.processor is None:
            raise RuntimeError("Processor not available – provide prepared inputs directly")
        prompt = self.processor.apply_chat_template(
            self._format_messages(messages, images), add_generation_prompt=True
        )
        return self.processor(text=prompt, images=images, return_tensors="np")

    def _format_messages(
        self,
        messages: List[Dict[str, Any]],
        images: List[Image.Image],
    ) -> List[Dict[str, Any]]:
        """Convert plain-string message content to SmolVLM's multimodal list format.

        SmolVLM's chat template requires content to be a list of typed parts so it
        can insert ``<image>`` tokens.  If the caller already provides the list form
        this is a no-op; plain strings are expanded to
        ``[{"type": "image"}, {"type": "text", "text": "..."}]``.
        """
        formatted: List[Dict[str, Any]] = []
        image_idx = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                # Already in multimodal list form — pass through unchanged.
                formatted.append(msg)
                continue
            # Plain string: inject an image placeholder for each remaining image.
            parts: List[Dict[str, Any]] = []
            if image_idx < len(images):
                parts.append({"type": "image"})
                image_idx += 1
            text = str(content).replace("<image>", "").strip()
            if text:
                parts.append({"type": "text", "text": text})
            formatted.append({"role": msg.get("role", "user"), "content": parts})
        return formatted

    def generate(self,
                 inputs: Dict[str, np.ndarray],
                 max_new_tokens: int = 100) -> str:
        """
        Generate text from prepared inputs (blocking).

        Args:
            inputs: Prepared inputs from prepare_inputs()
            max_new_tokens: Maximum tokens to generate

        Returns:
            Generated text string
        """
        tokens = list(self.generate_stream(inputs, max_new_tokens))
        if self.tokenizer:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return f"Generated {len(tokens)} tokens (no tokenizer available)"

    def generate_stream(self,
                        inputs: Dict[str, np.ndarray],
                        max_new_tokens: int = 100) -> Generator[int, None, None]:
        """
        Stream generation token by token.

        Args:
            inputs: Prepared inputs from prepare_inputs()
            max_new_tokens: Maximum tokens to generate

        Yields:
            Generated token IDs
        """
        batch_size = inputs['input_ids'].shape[0]
        past_key_values = self.decoder.initialize_past_key_values(batch_size)

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = np.cumsum(attention_mask, axis=-1)
        image_features = None  # computed once on first step

        for _ in range(max_new_tokens):
            inputs_embeds = self.embed.embed(input_ids)

            # Inject vision features once
            if image_features is None:
                image_features = self.vision.encode(
                    inputs['pixel_values'],
                    inputs['pixel_attention_mask'],
                )
                if self.image_token_id is not None:
                    image_mask = inputs['input_ids'] == self.image_token_id
                    if image_mask.any():
                        inputs_embeds[image_mask] = image_features.reshape(-1, image_features.shape[-1])

            logits, past_key_values = self.decoder.decode(
                inputs_embeds, attention_mask, position_ids, past_key_values
            )

            next_token_id = int(logits[:, -1].argmax(-1)[0])
            if next_token_id in self.eos_token_id:
                break

            yield next_token_id

            # Advance sequence
            input_ids = np.array([[next_token_id]], dtype=np.int64)
            attention_mask = np.ones_like(input_ids)
            position_ids = position_ids[:, -1:] + 1

    def __repr__(self) -> str:
        return (
            f"SmolVLMOnnx(model={self.model_path.name}, size={self.model_size}, "
            f"device={self.device}, layers={self.num_hidden_layers})"
        )
