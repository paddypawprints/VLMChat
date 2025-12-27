"""Text container for caching text data with format conversions."""
from typing import Optional, List, Dict, Any
from .item import CachedItem


class TextContainer(CachedItem):
    """
    Container for text data with format management.
    
    Supported formats:
    - "str": Python string (default)
    - "bytes": UTF-8 encoded bytes
    - "lines": List of strings (split by newline)
    
    All formats are lazily converted and cached on first access.
    """
    
    def __init__(self, text: str, cache_key: str, metadata: Optional[dict] = None):
        """
        Initialize text container.
        
        Args:
            text: The text string
            cache_key: Unique identifier for caching
            metadata: Optional metadata dict
        """
        super().__init__(cache_key)
        self._metadata = metadata or {}
        # Store initial format
        self._cached_formats = {}
        self._cached_formats["str"] = text
    
    def get(self, format: Optional[str] = "str") -> Any:
        """
        Get text in the specified format, with lazy conversion.
        
        Args:
            format: One of "str", "bytes", "lines" (defaults to "str")
            
        Returns:
            Text in the requested format
            
        Raises:
            ValueError: If format is not supported
        """
        # Default to str format if None
        if format is None:
            format = "str"
        
        # Return cached if available
        if format in self._cached_formats:
            return self._cached_formats[format]
        
        # Convert and cache
        if format == "str":
            # Get from any available format
            if "bytes" in self._cached_formats:
                result = self._cached_formats["bytes"].decode("utf-8")
            elif "lines" in self._cached_formats:
                result = "\n".join(self._cached_formats["lines"])
            else:
                raise ValueError(f"No source format available to convert to str")
            self._cached_formats["str"] = result
            return result
            
        elif format == "bytes":
            # Convert from str
            if "str" in self._cached_formats:
                result = self._cached_formats["str"].encode("utf-8")
            else:
                raise ValueError(f"str format required to convert to bytes")
            self._cached_formats["bytes"] = result
            return result
            
        elif format == "lines":
            # Convert from str
            if "str" in self._cached_formats:
                result = self._cached_formats["str"].split("\n")
            else:
                raise ValueError(f"str format required to convert to lines")
            self._cached_formats["lines"] = result
            return result
            
        else:
            raise ValueError(f"Unsupported text format: {format}. "
                           f"Supported formats: str, bytes, lines")
    
    def has_format(self, format: str) -> bool:
        """Check if format is cached."""
        return format in self._cached_formats
    
    def get_cached_formats(self) -> List[str]:
        """Return list of currently cached formats."""
        return list(self._cached_formats.keys())
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get item metadata without loading data."""
        return self._metadata
