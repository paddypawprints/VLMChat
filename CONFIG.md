# VLMChat Configuration System

This document describes the configuration system for the VLMChat application, which uses Pydantic for validation and type safety.

## Overview

The VLMChat application uses a centralized configuration system that supports:

- **Default values**: Sensible defaults for all settings
- **Configuration files**: JSON or YAML format configuration files
- **Environment variables**: Override settings via environment variables
- **Validation**: Type checking and value validation using Pydantic
- **Global access**: Singleton pattern for application-wide configuration

## Configuration Structure

The configuration is organized into four main sections:

### Model Configuration (`config.model`)

Controls the SmolVLM model behavior:

```json
{
  "model": {
    "model_path": "HuggingFaceTB/SmolVLM2-256M-Instruct",
    "max_new_tokens": 1024,
    "eos_token_id": 198,
    "use_onnx": true
  }
}
```

- **model_path**: Path to model (HuggingFace Hub or local path)
- **max_new_tokens**: Maximum tokens to generate (1-4096)
- **eos_token_id**: End-of-sequence token ID (≥0)
- **use_onnx**: Whether to use ONNX runtime for optimization

### Conversation Configuration (`config.conversation`)

Controls conversation history management:

```json
{
  "conversation": {
    "max_pairs": 10,
    "max_images": 1,
    "history_format": "xml",
    "word_limit": 15
  }
}
```

- **max_pairs**: Maximum conversation pairs to retain (1-1000)
- **max_images**: Maximum images in context (1-10)
- **history_format**: Format for history ("xml" or "minimal")
- **word_limit**: Word limit for minimal formatter (1-100)

### Logging Configuration (`config.logging`)

Controls application logging:

```json
{
  "logging": {
    "level": "INFO",
    "format": "%(asctime)s - %(levelname)s - %(message)s"
  }
}
```

- **level**: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
- **format**: Python logging format string

### Paths Configuration (`config.paths`)

Controls file paths and directories:

```json
{
  "paths": {
    "project_root": "/home/user/VLMChat/src",
    "coco_labels_path": "assets/coco_labels.txt",
    "captured_images_dir": "captured_images"
  }
}
```

- **project_root**: Application root directory (auto-detected if not set)
- **coco_labels_path**: Path to COCO labels file (relative to project_root)
- **captured_images_dir**: Directory for saving captured images

## Usage Methods

### 1. Using Default Configuration

The simplest way is to use default values:

```python
python src/main.py
```

### 2. Using Configuration Files

#### Create a default configuration file:

```bash
python src/main.py --create-config --config-output my_config.json
```

#### Use a configuration file:

```bash
python src/main.py --config my_config.json
```

#### Example configuration file (JSON):

```json
{
  "model": {
    "model_path": "my_custom_model",
    "max_new_tokens": 512,
    "use_onnx": false
  },
  "conversation": {
    "max_pairs": 20,
    "history_format": "minimal",
    "word_limit": 10
  },
  "logging": {
    "level": "DEBUG"
  }
}
```

#### Example configuration file (YAML):

```yaml
model:
  model_path: "my_custom_model"
  max_new_tokens: 512
  use_onnx: false

conversation:
  max_pairs: 20
  history_format: "minimal"
  word_limit: 10

logging:
  level: "DEBUG"
```

### 3. Using Environment Variables

Set environment variables with the prefix `VLMCHAT_`:

```bash
export VLMCHAT_MODEL_PATH="my_custom_model"
export VLMCHAT_MAX_PAIRS="20"
export VLMCHAT_LOG_LEVEL="DEBUG"
export VLMCHAT_USE_ONNX="false"

python src/main.py
```

**Available environment variables:**

- **Model**: `VLMCHAT_MODEL_PATH`, `VLMCHAT_MAX_NEW_TOKENS`, `VLMCHAT_EOS_TOKEN_ID`, `VLMCHAT_USE_ONNX`
- **Conversation**: `VLMCHAT_MAX_PAIRS`, `VLMCHAT_MAX_IMAGES`, `VLMCHAT_HISTORY_FORMAT`, `VLMCHAT_WORD_LIMIT`
- **Logging**: `VLMCHAT_LOG_LEVEL`, `VLMCHAT_LOG_FORMAT`
- **Paths**: `VLMCHAT_PROJECT_ROOT`, `VLMCHAT_COCO_LABELS_PATH`, `VLMCHAT_CAPTURED_IMAGES_DIR`

## Programmatic Usage

### Loading Configuration

```python
from src.config import load_config, get_config

# Load from file
config = load_config("my_config.json")

# Load from environment or defaults
config = load_config()

# Access global configuration
config = get_config()
```

### Accessing Configuration Values

```python
from src.config import get_config

config = get_config()

# Access nested values
model_path = config.model.model_path
max_pairs = config.conversation.max_pairs
log_level = config.logging.level
project_root = config.paths.project_root
```

### Creating Custom Configuration

```python
from src.config import VLMChatConfig, ModelConfig, ConversationConfig

# Create custom configuration
config = VLMChatConfig(
    model=ModelConfig(
        model_path="my_model",
        max_new_tokens=512
    ),
    conversation=ConversationConfig(
        max_pairs=5,
        history_format="minimal"
    )
)

# Save to file
config.save_to_file("custom_config.json")
```

## Validation

The configuration system automatically validates all values:

### Type Validation

```python
# This will raise a validation error
config = VLMChatConfig(
    model=ModelConfig(max_new_tokens="invalid")  # Should be int
)
```

### Range Validation

```python
# This will raise a validation error
config = VLMChatConfig(
    model=ModelConfig(max_new_tokens=0)  # Should be ≥1
)
```

### Format Validation

```python
# This will raise a validation error
config = VLMChatConfig(
    conversation=ConversationConfig(
        history_format="invalid"  # Should be "xml" or "minimal"
    )
)
```

## Testing Configuration

Use the provided test script to verify configuration functionality:

```bash
python test_config.py
```

This tests:
- Default configuration creation
- Configuration file creation and loading
- Environment variable loading
- Validation error handling
- Global configuration management

## Migration from Hardcoded Values

The configuration system replaces the following hardcoded values:

| Old Hardcoded Value | New Configuration Path |
|-------------------|------------------------|
| `"HuggingFaceTB/SmolVLM2-256M-Instruct"` | `config.model.model_path` |
| `1024` (max_new_tokens) | `config.model.max_new_tokens` |
| `198` (eos_token_id) | `config.model.eos_token_id` |
| `True` (use_onnx) | `config.model.use_onnx` |
| `10` (max_pairs) | `config.conversation.max_pairs` |
| `1` (max_images) | `config.conversation.max_images` |
| `HistoryFormat.XML` | `config.conversation.history_format` |
| `15` (word_limit) | `config.conversation.word_limit` |
| `logging.INFO` | `config.logging.level` |
| `'%(asctime)s - %(levelname)s - %(message)s'` | `config.logging.format` |
| `"assets/coco_labels.txt"` | `config.paths.coco_labels_path` |
| `"captured_images"` | `config.paths.captured_images_dir` |

## Dependencies

To use the configuration system, install the required dependencies:

```bash
pip install pydantic>=2.0.0
```

For YAML support:
```bash
pip install pyyaml
```

Or install all dependencies:
```bash
pip install -r requirements.txt
```

## Best Practices

1. **Use configuration files** for deployment-specific settings
2. **Use environment variables** for sensitive or environment-specific values
3. **Keep default values sensible** for development and testing
4. **Validate configuration early** in application startup
5. **Document custom configurations** for your specific use cases

## Error Handling

The configuration system provides detailed error messages:

```python
try:
    config = VLMChatConfig.load_from_file("invalid_config.json")
except ValueError as e:
    print(f"Configuration error: {e}")
```

Common error types:
- **FileNotFoundError**: Configuration file not found
- **ValueError**: Invalid configuration values or file format
- **ValidationError**: Pydantic validation failures