# Prompt Management Module

This module handles conversation history, prompt formatting, and context management for the VLMChat application.

## Components

### prompt.py
The `Prompt` class serves as a facade for all prompt and context-related operations, providing a unified interface over the history management system.

**Key Features:**
- Unified interface for prompt operations
- Integration with conversation history
- Current image and user input tracking

**Usage:**
```python
from prompt.prompt import Prompt
from prompt.history import HistoryFormat

# Initialize prompt manager
prompt = Prompt(
    max_pairs=10,
    max_images=1,
    history_format=HistoryFormat.XML
)

# Access components
history = prompt.history
current_image = prompt.current_image
user_input = prompt.user_input
```

### history.py
The `History` class manages conversation history with configurable formatting strategies and limits.

**Key Features:**
- Configurable conversation pair limits
- Multiple formatting strategies (XML, MINIMAL)
- Image context management
- Statistics and debugging information

**Usage:**
```python
from prompt.history import History, HistoryFormat

# Initialize history manager
history = History(
    max_pairs=10,
    max_images=1,
    history_format=HistoryFormat.XML
)

# Add conversation
history.add_conversation_pair("Hello", "Hi there!")
history.set_current_image(image)

# Get formatted output
formatted = history.get_formatted_history()
stats = history.get_stats()
```

### history_format_base.py
Abstract base class defining the interface for history formatting strategies.

**Interface:**
```python
from prompt.history_format_base import HistoryFormatBase

class CustomFormatter(HistoryFormatBase):
    @property
    def format_name(self) -> str:
        return "custom"

    def format_history(self, pairs) -> str:
        # Custom formatting logic
        return formatted_text
```

### history_format_xml.py
XML-based formatting strategy with structured tags for conversation history.

**Output Format:**
```xml
<conversation_history>
  <turn number="1">
    <request>What do you see?</request>
    <response>I can see a beautiful landscape.</response>
  </turn>
</conversation_history>
```

### history_format_minimal.py
Compact formatting strategy using abbreviations for token efficiency.

**Output Format:**
```
[H] Q: What do you see? A: I can see a beautiful landscape.
```

### history_format_factory.py
Factory class for creating appropriate formatting instances based on format type.

**Usage:**
```python
from prompt.history_format_factory import HistoryFormatFactory
from prompt.history import HistoryFormat

# Create formatter
formatter = HistoryFormatFactory.create_formatter(
    HistoryFormat.XML,
    {}  # Additional parameters
)
```

## Format Strategies

### XML Format
- **Purpose**: Structured, human-readable format
- **Use Case**: Detailed conversation analysis
- **Token Usage**: Higher (more verbose)
- **Structure**: Well-defined XML tags with turn numbers

### Minimal Format
- **Purpose**: Token-efficient compact format
- **Use Case**: Limited context window scenarios
- **Token Usage**: Lower (abbreviated)
- **Structure**: Compressed Q/A format with prefixes

## Configuration Options

### History Limits
- `max_pairs`: Maximum conversation pairs to retain (default: 10)
- `max_images`: Maximum images in context (currently limited to 1)

### Format Selection
- Choose between XML and MINIMAL formats
- Runtime format switching supported
- Factory pattern for easy extension

### Image Management
- Single image context (multimodal limitation)
- Automatic image replacement when new image loaded
- Image clearing with conversation history

## Usage Patterns

### Basic History Management
```python
# Initialize
history = History(max_pairs=5, history_format=HistoryFormat.XML)

# Add conversations
history.add_conversation_pair("Hello", "Hi there!")
history.add_conversation_pair("How are you?", "I'm doing well!")

# Get formatted output
formatted = history.get_formatted_history()
```

### Dynamic Format Switching
```python
# Change format at runtime
history.set_format(HistoryFormat.MINIMAL)
compact_history = history.get_formatted_history()

history.set_format(HistoryFormat.XML)
detailed_history = history.get_formatted_history()
```

### Statistics and Debugging
```python
# Get usage statistics
stats = history.get_stats()
print(f"Using {stats['pairs']} conversation pairs")
print(f"Format: {stats['format']}")

# Display human-readable history
print(str(history))  # Formatted for console display
```

## Dependencies

- **PIL**: Image handling and processing
- **Typing**: Type hints and annotations
- **Collections**: Deque for efficient conversation storage
- **Enum**: Format type definitions
- **Dataclasses**: Configuration structures