# Services Module

This module contains service classes that provide additional functionality to the VLMChat application, including metadata retrieval and augmentation services.

## Components

### rag_service.py
The `RAGService` (Retrieval Augmented Generation) class provides metadata retrieval functionality to enhance prompts with contextual information.

**Current Implementation:**
- Default metadata template system
- Placeholder for future vector database integration
- Metadata retrieval and update methods

**Key Features:**
- **Default Metadata**: Provides sample location and timing information
- **Retrieval Interface**: Designed for future vector-based retrieval
- **Update Capability**: Placeholder for dynamic metadata updates
- **Extensible Design**: Ready for integration with vector databases

**Usage:**
```python
from services.rag_service import RAGService

# Initialize service
rag = RAGService()

# Retrieve metadata (currently returns default template)
metadata = rag.retrieve_metadata(
    image=image,
    context="user query context"
)

# Update metadata with new information
updated_metadata = rag.update_metadata(
    location="Central Park",
    time="2:30 PM"
)
```

## Current Functionality

### Default Metadata Template
The service currently provides a default metadata structure:
```
place name: Central Park
city: Santa Clara
state: California
time of image capture: 2:30 PM
```

### Retrieval Methods
- `retrieve_metadata()`: Returns contextual metadata for prompts
- `update_metadata()`: Updates metadata with new information
- `default_metadata`: Property for accessing base template

## Future Enhancements

The RAGService is designed to support future integration with vector databases and advanced retrieval mechanisms:

### Planned Features
1. **Vector Database Integration**: Connect to vector stores for metadata retrieval
2. **Embedding Generation**: Create embeddings from images for similarity search
3. **Contextual Retrieval**: Retrieve relevant metadata based on image content
4. **Dynamic Updates**: Real-time metadata updates from external sources
5. **Custom Retrieval**: User-defined retrieval strategies

### Implementation Strategy
```python
# Future implementation concept
def retrieve_metadata(self, image=None, context="", **kwargs):
    """
    Enhanced retrieval with vector database support:
    1. Generate image embeddings using vision encoder
    2. Perform similarity search in metadata database
    3. Retrieve top-k most relevant metadata entries
    4. Format and return structured metadata
    """
    if self.vector_store:
        embeddings = self.encode_image(image)
        results = self.vector_store.similarity_search(embeddings)
        return self.format_metadata(results)
    else:
        return self.default_metadata
```

## Integration with VLMChat

The RAGService integrates with the main application to provide enhanced context:

1. **Prompt Augmentation**: Metadata is added to prompts for richer context
2. **Location Awareness**: Geographic and temporal information
3. **Contextual Understanding**: Enhanced model responses with metadata

## Configuration

The service can be configured for different use cases:
- **Default Mode**: Uses built-in metadata template
- **Custom Mode**: User-provided metadata sources
- **Vector Mode**: Integration with external vector databases

## Dependencies

- **Logging**: Error handling and debugging information
- **Typing**: Type hints for better code maintainability

## Usage Patterns

### Basic Metadata Retrieval
```python
# Simple metadata access
service = RAGService()
metadata = service.retrieve_metadata()
print(metadata)  # Default template
```

### Integration with Chat Application
```python
# In chat application context
rag_service = RAGService()
metadata = rag_service.retrieve_metadata(
    image=current_image,
    context=user_query
)

# Augment prompt with metadata
enhanced_prompt = f"{user_query}\n\nContext: {metadata}"
```

This service architecture provides a foundation for sophisticated metadata retrieval while maintaining simplicity for current use cases.