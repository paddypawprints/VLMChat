"""Data caching system with reference counting."""
from .item import CachedItem
from .types import CachedItemType, ITEM_TYPE_CLASSES
from .cache import ItemCache, StorageBackend, InProcessBackend
from .image import ImageContainer
from .embedding import EmbeddingContainer
from .text import TextContainer

__all__ = [
    "CachedItem",
    "CachedItemType",
    "ITEM_TYPE_CLASSES",
    "ItemCache",
    "StorageBackend",
    "InProcessBackend",
    "ImageContainer",
    "EmbeddingContainer",
    "TextContainer",
]
