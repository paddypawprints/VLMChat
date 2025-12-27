"""
Comprehensive unit tests for history formatting classes.

This module tests all history formatter implementations including
XML formatter, minimal formatter, and base class functionality.
"""

import pytest
from collections import deque
from abc import ABC

from src.prompt.history_format_base import HistoryFormatBase
from src.prompt.history_format_xml import HistoryFormatXML
from src.prompt.history_format_minimal import HistoryFormatMinimal
from .conftest import TestUtilities


class TestHistoryFormatBase:
    """Test the abstract base class for formatters."""

    @pytest.mark.unit
    def test_is_abstract_base_class(self):
        """Test that HistoryFormatBase is abstract."""
        assert issubclass(HistoryFormatBase, ABC)

        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            HistoryFormatBase()

    @pytest.mark.unit
    def test_abstract_methods_defined(self):
        """Test that abstract methods are properly defined."""
        abstract_methods = HistoryFormatBase.__abstractmethods__
        expected_methods = {'format_name', 'format_turn', 'format_history'}
        assert abstract_methods == expected_methods

    @pytest.mark.unit
    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement all abstract methods."""

        class IncompleteFormatter(HistoryFormatBase):
            @property
            def format_name(self):
                return "incomplete"
            # Missing format_turn and format_history

        with pytest.raises(TypeError):
            IncompleteFormatter()


class TestHistoryFormatXML:
    """Test XML formatting functionality."""

    @pytest.mark.unit
    def test_format_name(self, xml_formatter):
        """Test XML formatter name property."""
        assert xml_formatter.format_name == "xml"

    @pytest.mark.unit
    def test_format_single_turn_with_response(self, xml_formatter):
        """Test formatting single turn with both request and response."""
        result = xml_formatter.format_turn("Hello", "Hi there!")

        expected = "<user>Hello</user><assistant>Hi there!</assistant>"
        assert result == expected

    @pytest.mark.unit
    def test_format_single_turn_without_response(self, xml_formatter):
        """Test formatting single turn without response."""
        result = xml_formatter.format_turn("Hello")

        expected = "<user>Hello</user>"
        assert result == expected

    @pytest.mark.unit
    def test_format_single_turn_with_none_response(self, xml_formatter):
        """Test formatting single turn with None response."""
        result = xml_formatter.format_turn("Hello", None)

        expected = "<user>Hello</user>"
        assert result == expected

    @pytest.mark.unit
    def test_format_single_turn_with_empty_response(self, xml_formatter):
        """Test formatting single turn with empty string response."""
        result = xml_formatter.format_turn("Hello", "")

        expected = "<user>Hello</user>"
        assert result == expected

    @pytest.mark.unit
    def test_format_history_empty(self, xml_formatter, empty_deque):
        """Test formatting empty conversation history."""
        result = xml_formatter.format_history(empty_deque)
        assert result == ""

    @pytest.mark.unit
    def test_format_history_single_pair(self, xml_formatter):
        """Test formatting single conversation pair."""
        pairs = deque([("Hello", "Hi there!")])
        result = xml_formatter.format_history(pairs)

        expected = ("<conversation_history>"
                   "<turn_1>"
                   "<user>Hello</user>"
                   "<assistant>Hi there!</assistant>"
                   "</turn_1>"
                   "</conversation_history>")
        assert result == expected

    @pytest.mark.unit
    def test_format_history_multiple_pairs(self, xml_formatter, conversation_deque):
        """Test formatting multiple conversation pairs."""
        result = xml_formatter.format_history(conversation_deque)

        assert result.startswith("<conversation_history>")
        assert result.endswith("</conversation_history>")

        # Check each turn is properly formatted
        for i in range(1, len(conversation_deque) + 1):
            assert f"<turn_{i}>" in result
            assert f"</turn_{i}>" in result

        # Check specific content
        assert "<user>Hello</user>" in result
        assert "<assistant>Hi there!</assistant>" in result

    @pytest.mark.unit
    def test_format_history_with_missing_responses(self, xml_formatter):
        """Test formatting history with some missing responses."""
        pairs = deque([
            ("Question 1", "Answer 1"),
            ("Question 2", ""),
            ("Question 3", None),
            ("Question 4", "Answer 4")
        ])
        result = xml_formatter.format_history(pairs)

        # Should contain user questions but not empty responses
        assert "<user>Question 1</user>" in result
        assert "<assistant>Answer 1</assistant>" in result
        assert "<user>Question 2</user>" in result
        assert "<assistant></assistant>" not in result
        assert "<user>Question 3</user>" in result
        assert "<user>Question 4</user>" in result
        assert "<assistant>Answer 4</assistant>" in result

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_format_with_special_characters(self, xml_formatter, text_with_special_chars):
        """Test XML formatting with special characters."""
        result = xml_formatter.format_turn(text_with_special_chars, "Response")

        # Should contain the special characters as-is (not escaped)
        assert f"<user>{text_with_special_chars}</user>" in result
        assert "<assistant>Response</assistant>" in result

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_format_with_multiline_text(self, xml_formatter, multiline_text):
        """Test XML formatting with multiline text."""
        result = xml_formatter.format_turn(multiline_text, "Response")

        assert f"<user>{multiline_text}</user>" in result
        assert "\n" in result  # Newlines should be preserved

    @pytest.mark.unit
    def test_format_history_structure_validation(self, xml_formatter, sample_conversation_pairs):
        """Test that XML structure is valid and well-formed."""
        pairs = deque(sample_conversation_pairs)
        result = xml_formatter.format_history(pairs)

        TestUtilities.assert_xml_structure(result, len(sample_conversation_pairs))


class TestHistoryFormatMinimal:
    """Test minimal formatting functionality."""

    @pytest.mark.unit
    def test_default_initialization(self, minimal_formatter):
        """Test minimal formatter default initialization."""
        assert minimal_formatter.format_name == "minimal"
        assert minimal_formatter.word_limit == 15  # Default word limit

    @pytest.mark.unit
    def test_custom_word_limit_initialization(self, minimal_formatter_custom_limit):
        """Test minimal formatter with custom word limit."""
        assert minimal_formatter_custom_limit.format_name == "minimal"
        assert minimal_formatter_custom_limit.word_limit == 5

    @pytest.mark.unit
    def test_format_name(self, minimal_formatter):
        """Test minimal formatter name property."""
        assert minimal_formatter.format_name == "minimal"

    @pytest.mark.unit
    def test_word_limit_property(self, minimal_formatter_custom_limit):
        """Test word limit property."""
        assert minimal_formatter_custom_limit.word_limit == 5

    @pytest.mark.unit
    def test_format_single_turn_with_response(self, minimal_formatter):
        """Test formatting single turn with both request and response."""
        result = minimal_formatter.format_turn("Hello there", "Hi back!")

        expected = "U: Hello there | A: Hi back!"
        assert result == expected

    @pytest.mark.unit
    def test_format_single_turn_without_response(self, minimal_formatter):
        """Test formatting single turn without response."""
        result = minimal_formatter.format_turn("Hello there")

        expected = "U: Hello there"
        assert result == expected

    @pytest.mark.unit
    def test_format_single_turn_with_none_response(self, minimal_formatter):
        """Test formatting single turn with None response."""
        result = minimal_formatter.format_turn("Hello there", None)

        expected = "U: Hello there"
        assert result == expected

    @pytest.mark.unit
    def test_format_single_turn_with_empty_response(self, minimal_formatter):
        """Test formatting single turn with empty string response."""
        result = minimal_formatter.format_turn("Hello there", "")

        expected = "U: Hello there"
        assert result == expected

    @pytest.mark.unit
    def test_text_truncation_under_limit(self, minimal_formatter_custom_limit, short_text):
        """Test that short text is not truncated."""
        result = minimal_formatter_custom_limit._truncate_text(short_text)
        assert result == short_text
        assert "..." not in result

    @pytest.mark.unit
    def test_text_truncation_over_limit(self, minimal_formatter_custom_limit, long_text):
        """Test that long text is properly truncated."""
        result = minimal_formatter_custom_limit._truncate_text(long_text)

        words = result.split()
        # Should be exactly word_limit words plus ellipsis
        assert words[-1] == "..."
        assert len(words) == minimal_formatter_custom_limit.word_limit + 1
        assert result.endswith("...")

    @pytest.mark.unit
    def test_text_truncation_exact_limit(self, minimal_formatter_custom_limit):
        """Test text exactly at word limit is not truncated."""
        exact_text = " ".join([f"word{i}" for i in range(5)])  # Exactly 5 words
        result = minimal_formatter_custom_limit._truncate_text(exact_text)

        assert result == exact_text
        assert "..." not in result

    @pytest.mark.unit
    def test_format_turn_with_truncation(self, minimal_formatter_custom_limit, long_text):
        """Test format turn applies truncation."""
        result = minimal_formatter_custom_limit.format_turn(long_text, long_text)

        assert "..." in result
        assert "U:" in result
        assert "A:" in result
        assert "|" in result

    @pytest.mark.unit
    def test_format_history_empty(self, minimal_formatter, empty_deque):
        """Test formatting empty conversation history."""
        result = minimal_formatter.format_history(empty_deque)
        assert result == ""

    @pytest.mark.unit
    def test_format_history_single_pair(self, minimal_formatter):
        """Test formatting single conversation pair."""
        pairs = deque([("Hello", "Hi there!")])
        result = minimal_formatter.format_history(pairs)

        expected = "U: Hello | A: Hi there!"
        assert result == expected

    @pytest.mark.unit
    def test_format_history_multiple_pairs(self, minimal_formatter, conversation_deque):
        """Test formatting multiple conversation pairs."""
        result = minimal_formatter.format_history(conversation_deque)

        lines = result.split('\n')
        assert len(lines) == len(conversation_deque)

        for line in lines:
            assert "U:" in line
            if " | A:" in line:
                assert line.count("U:") == 1
                assert line.count("A:") == 1

        # Check specific content
        assert "U: Hello | A: Hi there!" in result

    @pytest.mark.unit
    def test_format_history_with_missing_responses(self, minimal_formatter):
        """Test formatting history with some missing responses."""
        pairs = deque([
            ("Question 1", "Answer 1"),
            ("Question 2", ""),
            ("Question 3", None),
            ("Question 4", "Answer 4")
        ])
        result = minimal_formatter.format_history(pairs)

        lines = result.split('\n')
        assert "U: Question 1 | A: Answer 1" in lines
        assert "U: Question 2" in lines  # No | A: for empty response
        assert "U: Question 3" in lines  # No | A: for None response
        assert "U: Question 4 | A: Answer 4" in lines

    @pytest.mark.unit
    def test_format_history_structure_validation(self, minimal_formatter, sample_conversation_pairs):
        """Test that minimal format structure is valid."""
        pairs = deque(sample_conversation_pairs)
        result = minimal_formatter.format_history(pairs)

        TestUtilities.assert_minimal_structure(result, len(sample_conversation_pairs))

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_truncation_with_empty_text(self, minimal_formatter, empty_text):
        """Test truncation with empty text."""
        result = minimal_formatter._truncate_text(empty_text)
        assert result == ""

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_truncation_with_single_word(self, minimal_formatter_custom_limit):
        """Test truncation with single word under limit."""
        result = minimal_formatter_custom_limit._truncate_text("word")
        assert result == "word"

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_format_with_special_characters(self, minimal_formatter, text_with_special_chars):
        """Test minimal formatting with special characters."""
        result = minimal_formatter.format_turn(text_with_special_chars, "Response")

        assert f"U: {text_with_special_chars}" in result
        assert "A: Response" in result

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_format_with_multiline_text(self, minimal_formatter, multiline_text):
        """Test minimal formatting with multiline text."""
        result = minimal_formatter.format_turn(multiline_text, "Response")

        assert f"U: {multiline_text}" in result
        assert "\n" in result  # Newlines should be preserved


class TestFormatterComparison:
    """Test comparison between different formatters."""

    @pytest.mark.unit
    def test_same_input_different_output(self, xml_formatter, minimal_formatter, conversation_deque):
        """Test that different formatters produce different outputs."""
        xml_result = xml_formatter.format_history(conversation_deque)
        minimal_result = minimal_formatter.format_history(conversation_deque)

        assert xml_result != minimal_result
        assert len(xml_result) > len(minimal_result)  # XML is more verbose

        # Both should contain the same basic content
        assert "Hello" in xml_result
        assert "Hello" in minimal_result

    @pytest.mark.unit
    def test_xml_more_verbose_than_minimal(self, xml_formatter, minimal_formatter):
        """Test that XML format is more verbose than minimal."""
        pairs = deque([("Question", "Answer")])

        xml_result = xml_formatter.format_history(pairs)
        minimal_result = minimal_formatter.format_history(pairs)

        # XML should have more characters due to tags
        assert len(xml_result) > len(minimal_result)

        # XML should have structured tags
        assert "<" in xml_result and ">" in xml_result
        assert "<" not in minimal_result and ">" not in minimal_result

    @pytest.mark.unit
    def test_both_formatters_preserve_content(self, xml_formatter, minimal_formatter):
        """Test that both formatters preserve original content."""
        request = "What do you see?"
        response = "I see a beautiful image."
        pairs = deque([(request, response)])

        xml_result = xml_formatter.format_history(pairs)
        minimal_result = minimal_formatter.format_history(pairs)

        # Both should contain original text
        assert request in xml_result
        assert response in xml_result
        assert request in minimal_result
        assert response in minimal_result


@pytest.mark.performance
class TestFormatterPerformance:
    """Test formatter performance characteristics."""

    @pytest.mark.unit
    def test_xml_formatter_large_history(self, xml_formatter):
        """Test XML formatter performance with large history."""
        # Create large conversation history
        pairs = deque([(f"Question {i}", f"Answer {i}") for i in range(100)])

        # Should complete quickly
        result = xml_formatter.format_history(pairs)

        assert len(result) > 0
        assert "<conversation_history>" in result
        assert result.count("<turn_") == 100

    @pytest.mark.unit
    def test_minimal_formatter_large_history(self, minimal_formatter):
        """Test minimal formatter performance with large history."""
        # Create large conversation history
        pairs = deque([(f"Question {i}", f"Answer {i}") for i in range(100)])

        # Should complete quickly
        result = minimal_formatter.format_history(pairs)

        lines = result.split('\n')
        assert len(lines) == 100
        assert all("U:" in line for line in lines)

    @pytest.mark.unit
    def test_truncation_performance(self, minimal_formatter_custom_limit):
        """Test truncation performance with very long text."""
        # Create very long text
        very_long_text = " ".join([f"word{i}" for i in range(1000)])

        # Should complete quickly
        result = minimal_formatter_custom_limit._truncate_text(very_long_text)

        assert result.endswith("...")
        words = result.split()
        assert len(words) == 6  # 5 words + "..."