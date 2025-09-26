"""
Comprehensive unit tests for the HistoryFormatFactory class.

This module tests the factory pattern implementation for creating
history formatters, including error handling and extensibility.
"""

import pytest
from typing import Type

from src.prompt.history import HistoryFormat
from src.prompt.history_format_factory import HistoryFormatFactory
from src.prompt.history_format_base import HistoryFormatBase
from src.prompt.history_format_xml import HistoryFormatXML
from src.prompt.history_format_minimal import HistoryFormatMinimal


class TestHistoryFormatFactory:
    """Test the HistoryFormatFactory class functionality."""

    @pytest.mark.unit
    def test_create_xml_formatter(self):
        """Test creating XML formatter through factory."""
        formatter = HistoryFormatFactory.create_formatter(HistoryFormat.XML)

        assert isinstance(formatter, HistoryFormatXML)
        assert isinstance(formatter, HistoryFormatBase)
        assert formatter.format_name == "xml"

    @pytest.mark.unit
    def test_create_minimal_formatter(self):
        """Test creating minimal formatter through factory."""
        formatter = HistoryFormatFactory.create_formatter(HistoryFormat.MINIMAL)

        assert isinstance(formatter, HistoryFormatMinimal)
        assert isinstance(formatter, HistoryFormatBase)
        assert formatter.format_name == "minimal"

    @pytest.mark.unit
    def test_create_xml_formatter_with_kwargs(self):
        """Test creating XML formatter with additional kwargs."""
        # XML formatter doesn't use kwargs, but should not fail
        formatter = HistoryFormatFactory.create_formatter(
            HistoryFormat.XML,
            unused_param="value"
        )

        assert isinstance(formatter, HistoryFormatXML)
        assert formatter.format_name == "xml"

    @pytest.mark.unit
    def test_create_minimal_formatter_with_kwargs(self):
        """Test creating minimal formatter with word limit kwargs."""
        formatter = HistoryFormatFactory.create_formatter(
            HistoryFormat.MINIMAL,
            word_limit=10
        )

        assert isinstance(formatter, HistoryFormatMinimal)
        assert formatter.format_name == "minimal"
        assert formatter.word_limit == 10

    @pytest.mark.unit
    def test_create_minimal_formatter_with_multiple_kwargs(self):
        """Test creating minimal formatter with multiple kwargs."""
        formatter = HistoryFormatFactory.create_formatter(
            HistoryFormat.MINIMAL,
            word_limit=20,
            unused_param="ignored"
        )

        assert isinstance(formatter, HistoryFormatMinimal)
        assert formatter.word_limit == 20

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_create_formatter_unsupported_type(self):
        """Test error handling for unsupported format types."""
        # Create a mock unsupported format
        class UnsupportedFormat:
            pass

        unsupported_format = UnsupportedFormat()

        with pytest.raises(ValueError) as exc_info:
            HistoryFormatFactory.create_formatter(unsupported_format)

        assert "Unsupported format type" in str(exc_info.value)
        assert "Supported formats" in str(exc_info.value)

    @pytest.mark.unit
    def test_get_supported_formats(self):
        """Test getting list of supported format types."""
        supported = HistoryFormatFactory.get_supported_formats()

        assert isinstance(supported, list)
        assert HistoryFormat.XML in supported
        assert HistoryFormat.MINIMAL in supported
        assert len(supported) == 2

    @pytest.mark.unit
    def test_format_map_completeness(self):
        """Test that format map contains all expected formats."""
        format_map = HistoryFormatFactory._format_map

        # Should contain all enum values
        expected_formats = {HistoryFormat.XML, HistoryFormat.MINIMAL}
        actual_formats = set(format_map.keys())
        assert actual_formats == expected_formats

        # Should map to correct classes
        assert format_map[HistoryFormat.XML] == HistoryFormatXML
        assert format_map[HistoryFormat.MINIMAL] == HistoryFormatMinimal

    @pytest.mark.unit
    def test_format_map_values_are_classes(self):
        """Test that format map values are classes, not instances."""
        format_map = HistoryFormatFactory._format_map

        for format_type, formatter_class in format_map.items():
            assert isinstance(formatter_class, type)
            assert issubclass(formatter_class, HistoryFormatBase)

    @pytest.mark.unit
    def test_factory_creates_new_instances(self):
        """Test that factory creates new instances each time."""
        formatter1 = HistoryFormatFactory.create_formatter(HistoryFormat.XML)
        formatter2 = HistoryFormatFactory.create_formatter(HistoryFormat.XML)

        # Should be different instances
        assert formatter1 is not formatter2
        # But same type
        assert type(formatter1) == type(formatter2)

    @pytest.mark.unit
    def test_factory_with_different_kwargs_creates_different_configs(self):
        """Test that different kwargs create differently configured instances."""
        formatter1 = HistoryFormatFactory.create_formatter(
            HistoryFormat.MINIMAL,
            word_limit=5
        )
        formatter2 = HistoryFormatFactory.create_formatter(
            HistoryFormat.MINIMAL,
            word_limit=15
        )

        assert formatter1.word_limit == 5
        assert formatter2.word_limit == 15

    @pytest.mark.unit
    def test_error_message_includes_supported_formats(self):
        """Test that error message includes list of supported formats."""
        class InvalidFormat:
            pass

        with pytest.raises(ValueError) as exc_info:
            HistoryFormatFactory.create_formatter(InvalidFormat())

        error_message = str(exc_info.value)
        assert "HistoryFormat.XML" in error_message
        assert "HistoryFormat.MINIMAL" in error_message


class TestFactoryExtensibility:
    """Test factory extensibility and modification."""

    @pytest.mark.unit
    def test_format_map_is_class_attribute(self):
        """Test that format map is a class attribute."""
        assert hasattr(HistoryFormatFactory, '_format_map')
        assert isinstance(HistoryFormatFactory._format_map, dict)

    @pytest.mark.unit
    def test_supported_formats_reflects_format_map(self):
        """Test that get_supported_formats reflects current format map."""
        original_map = HistoryFormatFactory._format_map.copy()
        supported = HistoryFormatFactory.get_supported_formats()

        # Should match format map keys
        assert set(supported) == set(original_map.keys())

    @pytest.mark.unit
    def test_classmethod_decorators(self):
        """Test that factory methods are class methods."""
        create_method = getattr(HistoryFormatFactory, 'create_formatter')
        supported_method = getattr(HistoryFormatFactory, 'get_supported_formats')

        # Should be bound to class, not instance
        assert hasattr(create_method, '__self__')
        assert hasattr(supported_method, '__self__')


class TestFactoryIntegration:
    """Test factory integration with other components."""

    @pytest.mark.unit
    def test_created_formatters_are_functional(self):
        """Test that created formatters are fully functional."""
        xml_formatter = HistoryFormatFactory.create_formatter(HistoryFormat.XML)
        minimal_formatter = HistoryFormatFactory.create_formatter(HistoryFormat.MINIMAL)

        # Test basic functionality
        turn_result_xml = xml_formatter.format_turn("Hello", "Hi!")
        turn_result_minimal = minimal_formatter.format_turn("Hello", "Hi!")

        assert turn_result_xml == "<user>Hello</user><assistant>Hi!</assistant>"
        assert turn_result_minimal == "U: Hello | A: Hi!"

    @pytest.mark.unit
    def test_created_formatters_work_with_history(self):
        """Test that created formatters work with History class."""
        from collections import deque

        formatter = HistoryFormatFactory.create_formatter(HistoryFormat.XML)
        pairs = deque([("Question", "Answer")])

        result = formatter.format_history(pairs)

        assert "<conversation_history>" in result
        assert "<user>Question</user>" in result
        assert "<assistant>Answer</assistant>" in result

    @pytest.mark.unit
    def test_factory_supports_all_history_format_enum_values(self):
        """Test that factory supports all values in HistoryFormat enum."""
        all_enum_values = list(HistoryFormat)
        supported_formats = HistoryFormatFactory.get_supported_formats()

        # Factory should support all enum values
        assert set(all_enum_values) == set(supported_formats)

        # Should be able to create formatter for each enum value
        for format_type in all_enum_values:
            formatter = HistoryFormatFactory.create_formatter(format_type)
            assert isinstance(formatter, HistoryFormatBase)


@pytest.mark.performance
class TestFactoryPerformance:
    """Test factory performance characteristics."""

    @pytest.mark.unit
    def test_factory_creation_performance(self):
        """Test that factory creation is fast."""
        # Create many formatters quickly
        formatters = []
        for _ in range(100):
            formatters.append(HistoryFormatFactory.create_formatter(HistoryFormat.XML))
            formatters.append(HistoryFormatFactory.create_formatter(HistoryFormat.MINIMAL))

        # Should have created all formatters
        assert len(formatters) == 200

        # All should be functional
        for formatter in formatters[:10]:  # Test first 10
            assert hasattr(formatter, 'format_name')
            assert callable(formatter.format_turn)

    @pytest.mark.unit
    def test_supported_formats_caching(self):
        """Test that get_supported_formats is efficient for multiple calls."""
        # Multiple calls should be fast
        results = []
        for _ in range(10):
            results.append(HistoryFormatFactory.get_supported_formats())

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result


@pytest.mark.edge_case
class TestFactoryEdgeCases:
    """Test factory edge cases and error conditions."""

    @pytest.mark.unit
    def test_none_format_type(self):
        """Test error handling for None format type."""
        with pytest.raises(ValueError):
            HistoryFormatFactory.create_formatter(None)

    @pytest.mark.unit
    def test_string_format_type(self):
        """Test error handling for string format type."""
        with pytest.raises(ValueError):
            HistoryFormatFactory.create_formatter("xml")

    @pytest.mark.unit
    def test_integer_format_type(self):
        """Test error handling for integer format type."""
        with pytest.raises(ValueError):
            HistoryFormatFactory.create_formatter(1)

    @pytest.mark.unit
    def test_empty_kwargs(self):
        """Test factory with empty kwargs dictionary."""
        formatter = HistoryFormatFactory.create_formatter(HistoryFormat.XML, **{})

        assert isinstance(formatter, HistoryFormatXML)
        assert formatter.format_name == "xml"

    @pytest.mark.unit
    def test_invalid_kwargs_ignored_gracefully(self):
        """Test that invalid kwargs are ignored gracefully."""
        # This should not raise an error, just ignore invalid kwargs
        formatter = HistoryFormatFactory.create_formatter(
            HistoryFormat.XML,
            invalid_param=123,
            another_invalid="value"
        )

        assert isinstance(formatter, HistoryFormatXML)

    @pytest.mark.unit
    def test_mixed_valid_invalid_kwargs(self):
        """Test mixing valid and invalid kwargs."""
        formatter = HistoryFormatFactory.create_formatter(
            HistoryFormat.MINIMAL,
            word_limit=8,  # valid
            invalid_param="ignored"  # invalid
        )

        assert isinstance(formatter, HistoryFormatMinimal)
        assert formatter.word_limit == 8