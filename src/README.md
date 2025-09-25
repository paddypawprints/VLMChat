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

The application can be configured through:
- Model paths and parameters in `ModelConfig`
- History limits and formatting in `History` class
- Camera settings via command-line arguments