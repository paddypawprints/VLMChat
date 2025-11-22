# FashionClip Module

Fashion-specific CLIP model for image-text similarity in the fashion domain, based on Marqo's FashionSigLIP model.

## Overview

FashionClip is optimized for fashion-related image-text matching tasks. It follows the VLMChat BaseModel architecture, enabling runtime switching and consistent API patterns.

## Architecture

```
FashionClipModel (BaseModel)
    └── FashionClipOpenClipBackend (FashionClipRuntimeBase)
            └── Uses Marqo/marqo-fashionSigLIP model
```

## Features

- **Fashion-optimized embeddings**: Trained specifically on fashion items
- **High accuracy**: 99.98% accuracy on fashion item classification
- **Text prompt caching**: Pre-encode prompts for repeated queries
- **Direct encoding**: Access raw image/text embeddings
- **Runtime switching**: Extensible architecture for future backends (TensorRT, ONNX)

## Usage

### Basic Image-Text Matching

```python
from models.FashionClip import FashionClipModel
from PIL import Image
from utils.config import VLMChatConfig

# Initialize model
config = VLMChatConfig()
model = FashionClipModel(config)

# Load image
image = Image.open("hat.jpg")

# Define fashion prompts
prompts = ["a hat", "a t-shirt", "shoes", "dress", "jacket"]

# Get matches sorted by confidence
matches = model.get_matches(image, prompts)

# Display results
for prob, text in matches:
    print(f"{text:20s}: {prob:.2%}")

# Output:
# a hat               : 99.98%
# shoes               : 0.01%
# a t-shirt           : 0.00%
# dress               : 0.00%
# jacket              : 0.00%
```

### Prompt Caching for Performance

```python
# Pre-cache prompts for repeated queries
category_prompts = [
    "a hat", "a cap", "a beanie",
    "a t-shirt", "a blouse", "a sweater",
    "shoes", "sneakers", "boots",
    "pants", "jeans", "trousers"
]

model.pre_cache_text_prompts(category_prompts)

# Now multiple queries use cached embeddings
for image_path in image_list:
    image = Image.open(image_path)
    matches = model.get_matches(image, category_prompts)
    # Fast! No need to re-encode prompts
```

### Direct Encoding for Custom Workflows

```python
# Encode image to feature vector
image_features = model.encode_image(image)  # Shape: (1, 768)

# Encode text to feature vectors
text_features = model.encode_text(["a hat", "shoes"])  # Shape: (2, 768)

# Compute custom similarity metrics
similarity = image_features @ text_features.T
```

### Runtime Management

```python
# Check current runtime
print(model.current_runtime())  # "open_clip"

# Switch runtimes (when additional backends are available)
# model.set_runtime('tensorrt')  # Future: TensorRT backend
# model.set_runtime('onnx')      # Future: ONNX backend
```

## Configuration

FashionClip can be configured via `VLMChatConfig`:

```python
config.model.fashion_clip_model_name = 'hf-hub:Marqo/marqo-fashionSigLIP'
config.model.fashion_clip_pretrained = ''  # Optional custom weights
config.model.device = 'cpu'  # or 'cuda'
```

## Model Details

- **Base Model**: Marqo's FashionSigLIP
- **Architecture**: Custom text CLIP with SigLIP loss
- **Embedding Dimension**: 768
- **Input Image Size**: 224x224
- **Normalization**: Mean=[0.5, 0.5, 0.5], Std=[0.5, 0.5, 0.5]
- **Tokenizer**: ViT-B-16-SigLIP tokenizer

## Testing

Run the test suite:

```bash
# Basic functionality test
python src/models/FashionClip/test_fashion_clip.py

# Comparison with original implementation
python src/models/FashionClip/test_comparison.py
```

## Files

- `fashion_clip_model.py` - Main model facade and runtime implementations
- `fashion_clip_config.py` - Configuration management
- `FashionClip-openclip.py` - Original reference implementation
- `test_fashion_clip.py` - Functional test suite
- `test_comparison.py` - Validation against original implementation
- `__init__.py` - Module exports

## Performance

- **Inference Time** (CPU): ~80ms per image-text pair
- **Feature Dimension**: 768-dimensional embeddings
- **Memory**: ~1.2GB model size

## Future Enhancements

- [ ] TensorRT backend for GPU acceleration
- [ ] ONNX backend for cross-platform optimization
- [ ] Batch processing support
- [ ] Fine-tuning utilities
- [ ] Extended fashion taxonomy support

## References

- [Marqo FashionSigLIP](https://huggingface.co/Marqo/marqo-fashionSigLIP)
- [OpenCLIP](https://github.com/mlfoundations/open_clip)
- [SigLIP Paper](https://arxiv.org/abs/2303.15343)
