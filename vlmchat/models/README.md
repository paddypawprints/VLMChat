# Models Module

This module contains model wrappers for vision-language processing, including SmolVLM for vision-language understanding, MobileCLIP for efficient image-text matching, and FashionClip for fashion-specific image-text similarity.

## Architecture

All models follow a common architecture pattern based on `BaseModel` and `BaseRuntime`:

- **BaseModel**: Abstract facade that manages configuration, metrics, and runtime switching
- **BaseRuntime**: Abstract interface for swappable backends (ONNX, TensorRT, OpenCLIP, etc.)
- Model-specific runtime interfaces extend BaseRuntime with domain-specific methods

This architecture enables:
- Runtime switching without code changes
- Consistent API across different backends
- Easy addition of new optimization backends

## Components

### FashionClip/
Fashion-specific CLIP model for image-text similarity in fashion domain.

#### fashion_clip_model.py
The `FashionClipModel` class provides a wrapper around Marqo's FashionSigLIP model with runtime switching support.

**Key Features:**
- Image-text similarity scoring for fashion items
- Pre-caching of text prompts for performance
- Normalized embeddings via OpenCLIP backend
- Runtime switching architecture (OpenCLIP currently available)
- Direct encoding methods for custom workflows

**Usage:**
```python
from models.FashionClip import FashionClipModel
from PIL import Image

# Initialize model
model = FashionClipModel(config)

# Get matches for an image
image = Image.open("hat.jpg")
prompts = ["a hat", "a t-shirt", "shoes"]
matches = model.get_matches(image, prompts)

# Results: [(0.95, "a hat"), (0.03, "shoes"), (0.02, "a t-shirt")]
for prob, text in matches:
    print(f"{text}: {prob:.2%}")

# Pre-cache prompts for repeated queries
model.pre_cache_text_prompts(prompts)

# Direct encoding for custom workflows
image_features = model.encode_image(image)
text_features = model.encode_text(prompts)
```

**Runtime Support:**
- `open_clip`: Default backend using OpenCLIP library with Marqo's FashionSigLIP
- `auto`: Automatically selects best available runtime

**Technical Details:**
- Model: `hf-hub:Marqo/marqo-fashionSigLIP`
- Normalized embeddings for cosine similarity
- Temperature-scaled softmax for probability distribution
- Feature caching for repeated prompt sets

#### fashion_clip_config.py
Configuration management for FashionClip model settings.

### MobileCLIP/
Efficient CLIP model optimized for mobile and edge devices.

#### clip_model.py
The `CLIPModel` class provides runtime-switchable CLIP implementations.

**Runtimes:**
- `open_clip`: OpenCLIP backend with MobileCLIP support
- `tensorrt`: TensorRT optimized backend (TODO)

**Usage:**
```python
from models.MobileClip.clip_model import CLIPModel

model = CLIPModel(config)
model.set_runtime('open_clip')

# Get similarity matches
matches = model.get_matches(image, ["person", "dog", "car"])
```

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