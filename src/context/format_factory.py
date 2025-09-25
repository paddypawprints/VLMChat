from typing import Type, Dict
from .base_context_format import BaseContextFormatting, ContextFormat
from .xml_context_formatting import XMLContextFormatting
from .minimal_context_formatting import MinimalContextFormatting

class ContextFormatFactory:
    """Factory for creating context formatters."""
    
    _format_map: Dict[ContextFormat, Type[BaseContextFormatting]] = {
        ContextFormat.XML: XMLContextFormatting,
        ContextFormat.MINIMAL: MinimalContextFormatting
    }
    
    @classmethod
    def create_formatter(cls, format_type: ContextFormat) -> BaseContextFormatting:
        """Create a new formatter instance of the specified type."""
        if format_type not in cls._format_map:
            raise ValueError(f"Unsupported format type: {format_type}")
            
        formatter_class = cls._format_map[format_type]
        return formatter_class()