"""Enumeration of cached item types for contract validation."""
from enum import Enum


class CachedItemType(Enum):
    """Types of cached items that can be stored in Context.data."""
    IMAGE = "image"                 # ImageContainer
    EMBEDDING = "embedding"         # EmbeddingContainer
    DETECTION = "detection"         # DetectionContainer (future)
    TEXT = "text"                   # TextContainer (future)
    TENSOR = "tensor"              # TensorContainer (future)


# Map to actual class names for instantiation
ITEM_TYPE_CLASSES = {
    CachedItemType.IMAGE: "ImageContainer",
    CachedItemType.EMBEDDING: "EmbeddingContainer",
    CachedItemType.DETECTION: "DetectionContainer",
    CachedItemType.TEXT: "TextContainer",
    CachedItemType.TENSOR: "TensorContainer",
}
