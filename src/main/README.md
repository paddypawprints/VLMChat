# Main Application Module

This module contains the main application entry point and chat interface for the VLMChat application.

## Components

### chat_application.py
The `SmolVLMChatApplication` class serves as the central coordinator for the entire application, integrating all components including the model, prompt management, image processing, and user interface.

**Key Features:**
- Model initialization with ONNX runtime support
- Interactive chat loop with command processing
- Image loading from URLs, files, and camera
- Conversation history management
- Context formatting and statistics

**Usage:**
```python
from main.chat_application import SmolVLMChatApplication

# Initialize with default SmolVLM model
app = SmolVLMChatApplication(
    model_path="HuggingFaceTB/SmolVLM2-256M-Instruct",
    use_onnx=True,
    max_pairs=10,
    history_format=HistoryFormat.XML
)

# Start interactive chat
app.run_interactive_chat()
```

**Available Commands:**
- `/load_url <url>` - Load image from URL
- `/load_file <path>` - Load image from local file
- `/camera` - Capture image from camera
- `/clear_context` - Clear conversation history
- `/show_context` - Display current history
- `/context_stats` - Show buffer statistics
- `/format <format>` - Change history format (xml|minimal)
- `/help` - Show help message
- `/quit` - Exit application

**Configuration Parameters:**
- `model_path`: HuggingFace model path or local path
- `use_onnx`: Enable ONNX runtime for optimization
- `max_pairs`: Maximum conversation pairs to retain
- `max_images`: Maximum images in context (currently 1)
- `history_format`: Conversation format (XML or MINIMAL)

**Dependencies:**
- SmolVLM model components from `models/`
- Prompt management from `prompt/`
- Image utilities from `utils/`
- Camera interface for Raspberry Pi IMX500