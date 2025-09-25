"""
Factory for creating conversation history formatters.

This module provides a factory class for creating different types of
conversation history formatters using the factory pattern, supporting
easy extensibility and configuration.
"""

from typing import Type, Dict
from .history_format_base import HistoryFormatBase
from .history import HistoryFormat
from .history_format_xml import HistoryFormatXML
from .history_format_minimal import HistoryFormatMinimal


class HistoryFormatFactory:
    """
    Factory for creating conversation history formatters.

    Implements the factory pattern to create appropriate formatter instances
    based on the requested format type, supporting easy extension with new
    formatting strategies.
    """

    _format_map: Dict[HistoryFormat, Type[HistoryFormatBase]] = {
        HistoryFormat.XML: HistoryFormatXML,
        HistoryFormat.MINIMAL: HistoryFormatMinimal
    }

    @classmethod
    def create_formatter(cls, format_type: HistoryFormat, **kwargs) -> HistoryFormatBase:
        """
        Create a new formatter instance of the specified type.

        Args:
            format_type: The type of formatter to create
            **kwargs: Additional configuration parameters passed to the formatter

        Returns:
            HistoryFormatBase: Instance of the requested formatter type

        Raises:
            ValueError: If the requested format type is not supported
        """
        if format_type not in cls._format_map:
            supported_formats = list(cls._format_map.keys())
            raise ValueError(f"Unsupported format type: {format_type}. "
                           f"Supported formats: {supported_formats}")

        formatter_class = cls._format_map[format_type]
        return formatter_class(**kwargs)

    @classmethod
    def get_supported_formats(cls) -> list[HistoryFormat]:
        """
        Get list of supported format types.

        Returns:
            list[HistoryFormat]: List of supported formatter types
        """
        return list(cls._format_map.keys())