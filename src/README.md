# VLMChat Source Code

This directory contains the source code for the SmolVLM (Small Vision Language Model) chat application - an interactive multimodal chatbot that can analyze images and engage in conversations about them.

## Overview

VLMChat is built around the HuggingFace SmolVLM model with ONNX runtime optimization for efficient inference on edge devices like Raspberry Pi. The application supports real-time image capture via IMX500 camera, conversation history management, and flexible prompt formatting.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

- **main/**: Application entry point and main chat interface
- **models/**: SmolVLM model wrapper and configuration
- **prompt/**: Conversation history and prompt management
- **services/**: RAG service for metadata retrieval
- **utils/**: Utility functions for image processing and camera interface
- **tests/**: Unit tests for the application components

## Key Features

- **Vision-Language Model**: Powered by SmolVLM for multimodal understanding
- **ONNX Optimization**: Optional ONNX runtime for faster inference
- **Camera Integration**: Direct integration with Raspberry Pi IMX500 camera
- **Conversation Management**: Configurable history limits and formatting
- **Image Loading**: Support for URLs, local files, and camera capture
- **Interactive Chat**: Command-line interface with slash commands

## Quick Start

```bash
# Run the application
python3 src/main.py

# Check ONNX model status
python3 src/main.py --onnx-info

# Load an image and start chatting
/load_url https://example.com/image.jpg
What do you see in this image?
```

## Dependencies

- PyTorch and Transformers for model inference
- PIL for image processing
- ONNX Runtime for optimization (optional)
- Picamera2 for Raspberry Pi camera integration
- Requests for URL image loading

## Configuration

The application uses a centralized Pydantic-based configuration system that supports:

### Configuration Methods
- **Default values**: Sensible defaults for all settings
- **Configuration files**: JSON or YAML format configuration files
- **Environment variables**: Override settings with `VLMCHAT_` prefixed variables
- **Command-line options**: Specify config file or create default config

### Quick Configuration
```bash
# Create default configuration file
python src/main.py --create-config

# Use custom configuration file
python src/main.py --config my_config.json

# Use environment variables
export VLMCHAT_MODEL_PATH="my_custom_model"
export VLMCHAT_MAX_PAIRS="20"
export VLMCHAT_LOG_LEVEL="DEBUG"
python src/main.py
```

### Configuration Sections
- **Model**: Model paths, token limits, ONNX settings
- **Conversation**: History limits, formatting, word limits
- **Logging**: Log levels and format strings
- **Paths**: File paths and directories

For detailed configuration documentation, see [CONFIG.md](../CONFIG.md).

## Testing

The codebase includes a comprehensive test suite ensuring reliability, performance, and maintainability across all components.

### Test Suite Overview

The test suite is organized into focused test modules:

- **`tests/test_prompt/`**: Complete test coverage for the prompt module
  - Unit tests for conversation history management
  - Integration tests for component interactions
  - Performance tests for scalability
  - Edge case tests for robustness

### Running Tests

#### Quick Start
```bash
# Run all prompt module tests
python -m pytest src/tests/test_prompt/

# Using the provided test runner
python run_prompt_tests.py
```

#### Test Categories
```bash
# Unit tests only (fast)
python run_prompt_tests.py --unit

# Integration tests
python run_prompt_tests.py --integration

# Performance tests
python run_prompt_tests.py --performance

# Edge case and boundary tests
python run_prompt_tests.py --edge-case

# Quick test suite
python run_prompt_tests.py --quick

# Full test suite with coverage
python run_prompt_tests.py --full
```

#### Advanced Testing Options
```bash
# Run with coverage report
python run_prompt_tests.py --coverage

# Run specific test file
python run_prompt_tests.py --test-file test_history.py

# Parallel execution (requires pytest-xdist)
python run_prompt_tests.py --parallel 4

# Verbose output
python run_prompt_tests.py --verbose

# Custom pytest arguments
python run_prompt_tests.py --pytest-args "-k conversation"
```

### Test Coverage

The test suite provides comprehensive coverage:

#### Prompt Module (150+ tests)
- ✅ **History Management**: Conversation storage, limits, clearing
- ✅ **Image Handling**: Image setting, validation, memory management
- ✅ **Format Management**: XML/minimal formatting, runtime switching
- ✅ **Facade Pattern**: Prompt interface, delegation, consistency
- ✅ **Factory Pattern**: Formatter creation, configuration, extensibility
- ✅ **Integration**: End-to-end workflows, component coordination
- ✅ **Performance**: Scalability, memory efficiency, large conversations
- ✅ **Edge Cases**: Unicode support, error recovery, boundary conditions

#### Performance Benchmarks
- **1000 conversations**: < 1 second to process
- **Format switching**: < 1 second for 100 operations
- **Large text formatting**: < 0.1 second per operation
- **Memory usage**: Bounded by conversation limits

### Test Requirements

#### Core Requirements
```bash
# Essential packages
pip install pytest pillow
```

#### Optional Enhancements
```bash
# Coverage reporting
pip install pytest-cov

# Parallel test execution
pip install pytest-xdist

# Test timeouts
pip install pytest-timeout

# Performance benchmarking
pip install pytest-benchmark
```

### Test Configuration

The test suite uses pytest with custom configuration:

- **Markers**: `unit`, `integration`, `performance`, `edge_case`
- **Coverage**: HTML and terminal reports available
- **Fixtures**: Comprehensive shared test data and utilities
- **Parallel**: Supports parallel execution for faster runs

### Development Testing

For development workflows:

```bash
# Watch mode (requires pytest-watch)
ptw src/tests/test_prompt/

# Test specific functionality
python -m pytest src/tests/test_prompt/test_history.py::TestHistoryInitialization

# Debug failing tests
python -m pytest src/tests/test_prompt/ --pdb

# Generate coverage report
python -m pytest src/tests/test_prompt/ --cov=src.prompt --cov-report=html
```

### Continuous Integration

The test suite is designed for CI/CD integration:

- **Exit codes**: Proper exit codes for CI systems
- **Markers**: Allow selective test execution
- **Reports**: Generate coverage and performance reports
- **Timeouts**: Prevent hanging tests in CI environments

For more detailed testing information, see the [test suite documentation](tests/test_prompt/README.md).