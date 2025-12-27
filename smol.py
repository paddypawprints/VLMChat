#!/usr/bin/env python3
"""
Simple SmolVLM text-only generator with interactive mode.

Usage:
    python smol.py [--model 256M|500M]
    
    Then type prompts interactively (Ctrl+C or 'exit' to quit)
"""
import sys
import logging
from pathlib import Path
import argparse

# Setup logging
logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Direct imports to avoid pipeline dependencies
import numpy as np
from PIL import Image

# Import backends directly
from vlmchat.pipeline.models.smolvlm_vision_onnx import SmolVLMVisionOnnx
from vlmchat.pipeline.models.smolvlm_embed_onnx import SmolVLMEmbedOnnx
from vlmchat.pipeline.models.smolvlm_decoder_onnx import SmolVLMDecoderOnnx
from typing import List, Dict, Generator

try:
    from transformers import AutoProcessor, AutoConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("ERROR: transformers not available")
    sys.exit(1)

def create_dummy_image():
    """Create a 1x1 dummy image for text-only mode."""
    return Image.new('RGB', (1, 1), color='black')

class SimpleSmolVLM:
    """Minimal SmolVLM wrapper for text-only generation."""
    
    def __init__(self, model_path: str, model_size: str = "256M"):
        self.model_path = Path(model_path)
        self.model_size = model_size
        
        # Determine HuggingFace model name based on size
        hf_model_name = f"HuggingFaceTB/SmolVLM2-{model_size}-Instruct"
        
        # Load processor and config
        self.config = AutoConfig.from_pretrained(hf_model_name)
        self.processor = AutoProcessor.from_pretrained(hf_model_name)
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
        else:
            self.eos_token_id = []
        
        # Add tokenizer EOS if different
        if self.tokenizer.eos_token_id not in self.eos_token_id:
            self.eos_token_id.append(self.tokenizer.eos_token_id)
        
        # Load backends
        self.vision = SmolVLMVisionOnnx(str(self.model_path / "vision_encoder.onnx"), device="cuda")
        self.embed = SmolVLMEmbedOnnx(str(self.model_path / "embed_tokens.onnx"), device="cuda")
        self.decoder = SmolVLMDecoderOnnx(
            str(self.model_path / "decoder_model_merged.onnx"),
            num_hidden_layers=self.num_hidden_layers,
            num_key_value_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            device="cpu"
        )
        
        # Cache for image features (computed once, reused across prompts)
        self.cached_image_features = None
    
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode image once and cache the features."""
        # Process image to get pixel values
        pixel_inputs = self.processor(images=[image], return_tensors="np")
        pixel_values = pixel_inputs['pixel_values']
        pixel_attention_mask = pixel_inputs['pixel_attention_mask']
        
        # Encode vision features
        image_features = self.vision.encode(pixel_values, pixel_attention_mask)
        self.cached_image_features = image_features
        return image_features
    
    def prepare_inputs(self, messages: List[Dict[str, str]], images: List[Image.Image], text_only: bool = False) -> Dict[str, np.ndarray]:
        """Prepare inputs for generation."""
        # Check if user prompt contains <image> token
        user_content = messages[0]["content"] if messages else ""
        has_image_token = "<image>" in user_content
        
        if text_only and not has_image_token:
            # Pure text mode: no image token in prompt
            prompt = ""
            for msg in messages:
                if msg["role"] == "user":
                    prompt += msg["content"]
                elif msg["role"] == "assistant":
                    prompt += msg["content"]
            
            # Add generation prompt manually
            prompt = f"User: {prompt}\n\nAssistant:"
            
            # Tokenize without images
            inputs = self.tokenizer(prompt, return_tensors="np", padding=True)
            
            # Create dummy pixel values/masks with correct shape (batch, num_images, channels, height, width)
            inputs['pixel_values'] = np.zeros((1, 1, 3, 384, 384), dtype=np.float32)
            inputs['pixel_attention_mask'] = np.ones((1, 384, 384), dtype=np.bool_)
        else:
            # Image mode: convert to proper message format with image content type
            # The processor expects messages with content=[{"type": "image"}, {"type": "text", ...}]
            modified_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    # Remove <image> token from text if present (we'll add it via content type)
                    text_content = msg["content"].replace("<image>", "").strip()
                    # Create structured content with image first, then text
                    modified_messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": text_content}
                        ]
                    })
                else:
                    modified_messages.append(msg)
            
            prompt = self.processor.apply_chat_template(modified_messages, add_generation_prompt=True)
            inputs = self.processor(text=prompt, images=images, return_tensors="np")
        
        return inputs, prompt
    
    def generate_stream(self, inputs: Dict[str, np.ndarray], max_new_tokens: int = 256, text_only: bool = False) -> Generator[int, None, None]:
        """Stream generation token by token."""
        batch_size = inputs['input_ids'].shape[0]
        
        # Initialize past key-values
        past_key_values = self.decoder.initialize_past_key_values(batch_size)
        
        # Use cached image features if available, otherwise encode
        image_features = None
        if not text_only:
            if self.cached_image_features is not None:
                image_features = self.cached_image_features
            else:
                image_features = self.vision.encode(
                    inputs['pixel_values'],
                    inputs['pixel_attention_mask']
                )
        
        # Prepare initial inputs
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        position_ids = np.cumsum(attention_mask, axis=-1)
        
        # Generation loop
        for i in range(max_new_tokens):
            # Get token embeddings
            inputs_embeds = self.embed.embed(input_ids)
            
            # Inject image features on first step (only if we have them)
            if i == 0 and image_features is not None and self.image_token_id is not None:
                # Replace image token embeddings with vision features
                image_mask = input_ids == self.image_token_id
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

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='SmolVLM Text Generator')
    parser.add_argument('--model', type=str, default='256M', choices=['256M', '500M'],
                        help='Model size to use (default: 256M)')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to image file (optional, for vision+text mode)')
    args = parser.parse_args()
    
    print(f"\n{'='*70}")
    print(f"SmolVLM Text Generator ({args.model})")
    print(f"{'='*70}")
    
    # Initialize model once
    print(f"\nLoading SmolVLM-{args.model} model...")
    model_path = Path.home() / "onnx" / f"SmolVLM2-{args.model}-Instruct"
    
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        print(f"Please ensure the SmolVLM2-{args.model}-Instruct model is downloaded to ~/onnx/")
        return
    
    model = SimpleSmolVLM(str(model_path), model_size=args.model)
    print("✓ Model loaded")
    
    # Load and encode image if provided (only once!)
    image = None
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return
        from PIL import Image
        image = Image.open(image_path).convert('RGB')
        print(f"✓ Image loaded: {image_path.name} ({image.size})")
        print("  Encoding image features...")
        model.encode_image(image)
        print(f"✓ Image features cached (use <image> in prompts to reference)")
    else:
        image = create_dummy_image()
        print("✓ Text-only mode (no image)")
    
    print("\nEnter prompts (Ctrl+C or Ctrl+D to exit)")
    print("="*70)
    
    # Interactive loop
    try:
        while True:
            # Get prompt from user
            try:
                prompt = input("\nYou: ").strip()
            except EOFError:
                print("\n\nExiting...")
                break
            
            if not prompt:
                continue
            
            # Check if prompt contains <image> token
            has_image_token = "<image>" in prompt
            # Use vision mode if we have cached features (unless user explicitly wants text-only)
            use_vision = model.cached_image_features is not None
            
            # Prepare inputs
            messages = [{"role": "user", "content": prompt}]
            inputs, formatted_prompt = model.prepare_inputs(messages, [image], text_only=not use_vision)
            
            # Generate response
            print("\nAssistant: ", end='', flush=True)
            
            # Stream generation token by token
            tokens = []
            for token_id in model.generate_stream(inputs, max_new_tokens=256, text_only=not use_vision):
                tokens.append(token_id)
                
                # Decode and print token
                if model.tokenizer:
                    token_text = model.tokenizer.decode([token_id], skip_special_tokens=False)
                    print(token_text, end='', flush=True)
            
            print()  # Newline after response
            
    except KeyboardInterrupt:
        print("\n\nExiting...")
    
    print(f"\n{'='*70}")
    print("Goodbye!")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
