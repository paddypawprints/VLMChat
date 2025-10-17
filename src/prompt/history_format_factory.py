"""
Factory for creating conversation history formatters.

This module provides a factory class for creating different types of
conversation history formatters using the factory pattern, supporting
easy extensibility and configuration.
"""

from typing import Type, Dict
from .history_format_base import HistoryFormatBase
from .history_format import HistoryFormat
from .history_format_xml import HistoryFormatXML
from .history_format_minimal import HistoryFormatMinimal


class HistoryFormatFactory:
    """
    Factory for creating conversation history formatters.

    Implements the factory pattern to create appropriate formatter instances
    based on the requested format type, supporting easy extension with new
    formatting strategies.
    """

    # Only include the supported formats used across the codebase and tests.
    _format_map: Dict[HistoryFormat, Type[HistoryFormatBase]] = {
        HistoryFormat.XML: HistoryFormatXML,
        HistoryFormat.MINIMAL: HistoryFormatMinimal,
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
        # Reject plain string inputs (tests expect ValueError for string types).
        if isinstance(format_type, str):
            raise ValueError("format_type must be a HistoryFormat enum, not a string")

        # Accept either the HistoryFormat enum or a compatible enum from another
        # import path. Resolve by comparing `.value` when direct key lookup fails.
        if format_type not in cls._format_map:
            # try to resolve by .value or by string
            val = None
            try:
                val = format_type.value
            except Exception:
                val = str(format_type)

            resolved = None
            for k in cls._format_map.keys():
                if getattr(k, 'value', None) == val or str(k) == val:
                    resolved = k
                    break

            if resolved is None:
                supported_formats = list(cls._format_map.keys())
                raise ValueError(f"Unsupported format type: {format_type}. Supported formats: {supported_formats}")
            formatter_class = cls._format_map[resolved]
        else:
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