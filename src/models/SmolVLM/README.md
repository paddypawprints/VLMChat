# SmolVLM Model Implementation

This directory contains the complete implementation of the SmolVLM (Small Vision Language Model) with ONNX runtime optimization.

## Components

### smol_vlm_model.py
Core model wrapper that provides both standard transformers and ONNX runtime inference paths.

**Architecture:**
- **Vision Encoder**: Processes PIL images into feature representations
- **Text Embedder**: Converts token IDs to embeddings
- **Decoder**: Transformer decoder with attention mechanism
- **Tokenizer**: Handles text tokenization and decoding

**Inference Modes:**
1. **ONNX Runtime** (Preferred): Optimized inference with separate sessions
2. **Transformers** (Fallback): Standard HuggingFace generation

### model_config.py
Configuration dataclass for model parameters and settings.

**Key Settings:**
- Model path (HuggingFace Hub or local)
- Token generation limits
- Special token definitions
- End-of-sequence handling

### response_generator.py
Orchestrates text generation with streaming support and error handling.

**Generation Process:**
1. Input preparation (text + images)
2. Model inference (ONNX or transformers)
3. Token streaming for real-time display
4. Response post-processing

## Usage Examples

### Basic Model Usage
```python
from models.SmolVLM.smol_vlm_model import SmolVLMModel
from models.SmolVLM.model_config import ModelConfig

# Configure model
config = ModelConfig(
    model_path="HuggingFaceTB/SmolVLM2-256M-Instruct",
    max_new_tokens=512
)

# Initialize with ONNX support
model = SmolVLMModel(config, use_onnx=True)

# Format conversation
messages = model.get_messages(prompt_object)
inputs = model.prepare_inputs(messages, [image])

# Generate response
if model.use_onnx:
    for token in model.generate_onnx(inputs):
        print(token, end='', flush=True)
else:
    response = model.generate_transformers(inputs)
    print(response)
```

### Response Generation
```python
from models.SmolVLM.response_generator import ResponseGenerator

# Initialize generator
generator = ResponseGenerator(model)

# Generate with streaming
response = generator.generate_response(
    messages=messages,
    images=[image],
    stream_output=True
)
```

## Technical Implementation

### ONNX Runtime Setup
The model exports three separate ONNX models:
- `vision_encoder.onnx`: Image feature extraction
- `embed_tokens.onnx`: Token embedding layer
- `decoder_model_merged.onnx`: Transformer decoder

### Key-Value Caching
Implements efficient attention caching for autoregressive generation:
- Maintains past key-values across generation steps
- Reduces redundant computation for long sequences
- Supports batch processing

### Image Integration
- Processes PIL images through vision encoder
- Replaces image token embeddings with visual features
- Maintains spatial relationships in multimodal context

## Performance Characteristics

- **Memory Usage**: bfloat16 precision reduces memory footprint
- **Inference Speed**: ONNX runtime provides 2-3x speedup on CPU
- **Streaming**: Token-by-token generation for responsive UI
- **Fallback**: Automatic fallback to transformers if ONNX fails

## Dependencies

- **PyTorch**: Core tensor operations
- **Transformers**: Model loading and tokenization
- **ONNX Runtime**: Optimized inference engine
- **PIL**: Image processing
- **NumPy**: Array operations for ONNX interface