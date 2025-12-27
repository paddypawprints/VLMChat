"""
Base class for cached pipeline items.

All items in Context.data are CachedItem instances that provide
access to underlying data through format-specific accessors.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class CachedItem(ABC):
    """
    Base class for cached items with multi-format support.
    
    Each item type (ImageContainer, EmbeddingContainer, etc.) manages
    its own format conversions and caching strategy.
    """
    
    def __init__(self, cache_key: str):
        self._cache_key = cache_key
    
    @abstractmethod
    def get(self, format: Optional[str] = None) -> Any:
        """
        Get item data in specified format.
        
        Args:
            format: Desired format (None = primary/default format)
        
        Returns:
            Item data in requested format
        """
        pass
    
    def __call__(self, format: Optional[str] = None) -> Any:
        """Shorthand for get()."""
        return self.get(format)
    
    @abstractmethod
    def has_format(self, format: str) -> bool:
        """Check if format is currently cached."""
        pass
    
    @abstractmethod
    def get_cached_formats(self) -> List[str]:
        """Get list of currently cached formats."""
        pass
    
    @property
    def cache_key(self) -> str:
        """Unique cache identifier."""
        return self._cache_key
    
    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """Get item metadata without loading data."""
        pass
    
    def __repr__(self) -> str:
        formats = ', '.join(self.get_cached_formats())
        return f"{self.__class__.__name__}({self._cache_key[:8]}..., formats=[{formats}])"
