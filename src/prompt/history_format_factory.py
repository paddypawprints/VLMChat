from typing import Type, Dict
from .history_format_base import HistoryFormatBase, HistoryFormat
from .history_format_xml import HistoryFormatXML
from .history_format_minimal import HistoryFormatMinimal

class HistoryFormatFactory:
    """Factory for creating context formatters."""
    
    _format_map: Dict[HistoryFormat, Type[HistoryFormatBase]] = {
        HistoryFormat.XML: HistoryFormatXML,
        HistoryFormat.MINIMAL: HistoryFormatMinimal
    }
    
    @classmethod
    def create_formatter(cls, format_type: HistoryFormat, **kwargs) -> HistoryFormatBase:
        """Create a new formatter instance of the specified type."""
        if format_type not in cls._format_map:
            raise ValueError(f"Unsupported format type: {format_type}")
            
        formatter_class = cls._format_map[format_type]
        return formatter_class(**kwargs)