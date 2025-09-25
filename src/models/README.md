# Models Module

This module contains the SmolVLM model wrapper and related components for vision-language processing.

## Components

### SmolVLM/
Contains the complete SmolVLM model implementation with ONNX runtime support.

#### smol_vlm_model.py
The `SmolVLMModel` class provides a comprehensive wrapper around the HuggingFace SmolVLM model with optional ONNX runtime optimization.

**Key Features:**
- HuggingFace Transformers integration
- ONNX runtime support for faster inference
- Streaming text generation
- Image and text input processing
- Autoregressive generation with key-value caching

**Usage:**
```python
from models.SmolVLM.smol_vlm_model import SmolVLMModel
from models.SmolVLM.model_config import ModelConfig

config = ModelConfig(model_path="HuggingFaceTB/SmolVLM2-256M-Instruct")
model = SmolVLMModel(config, use_onnx=True)

# Prepare inputs
inputs = model.prepare_inputs(messages, images)

# Generate with ONNX (streaming)
for token in model.generate_onnx(inputs):
    print(token, end='', flush=True)

# Generate with transformers (fallback)
response = model.generate_transformers(inputs)
```

#### model_config.py
The `ModelConfig` dataclass holds all configuration parameters for model initialization.

**Configuration Options:**
- `model_path`: Path to model (HuggingFace Hub or local)
- `max_new_tokens`: Maximum tokens to generate (default: 1024)
- `eos_token_id`: End-of-sequence token ID (default: 198)
- `special_tokens`: Dictionary of special token mappings

#### response_generator.py
The `ResponseGenerator` class handles text generation orchestration and streaming.

**Features:**
- Automatic fallback from ONNX to transformers
- Streaming response generation
- Error handling and logging
- Token-by-token output for real-time display

**Technical Details:**
- **Model Format**: bfloat16 precision for memory efficiency
- **ONNX Sessions**: Separate sessions for vision encoder, embeddings, and decoder
- **Key-Value Caching**: Maintains attention context for autoregressive generation
- **Image Processing**: Converts PIL images to model input tensors
- **Chat Templates**: Formats conversation messages for the model

**Dependencies:**
- PyTorch for tensor operations
- Transformers for model loading
- ONNX Runtime for optimized inference
- PIL for image processing
- NumPy for array operations

**Performance:**
- ONNX runtime provides significant speedup on CPU inference
- Streaming generation enables responsive user experience
- Memory-efficient attention caching for long conversations