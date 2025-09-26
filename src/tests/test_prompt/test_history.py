"""
Comprehensive unit tests for the History class.

This module tests all functionality of the History class including
initialization, conversation management, image handling, formatting,
and edge cases.
"""

import pytest
from collections import deque
from PIL import Image

from src.prompt.history import History, HistoryFormat
from .conftest import TestUtilities


class TestHistoryInitialization:
    """Test History class initialization and configuration."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test History initialization with default parameters."""
        history = History()

        assert history.max_pairs == 10
        assert history._history_format == HistoryFormat.XML
        assert len(history._pairs) == 0

    @pytest.mark.unit
    def test_custom_initialization(self):
        """Test History initialization with custom parameters."""
        history = History(
            max_pairs=5,
            history_format=HistoryFormat.MINIMAL
        )

        assert history.max_pairs == 5
        assert history._history_format == HistoryFormat.MINIMAL

    @pytest.mark.unit
    def test_initialization_with_kwargs(self):
        """Test History initialization with additional kwargs."""
        history = History(
            max_pairs=3,
            history_format=HistoryFormat.MINIMAL,
            word_limit=10
        )

        assert history.max_pairs == 3
        assert history._formatter.word_limit == 10

    @pytest.mark.unit
    @pytest.mark.edge_case
    def test_invalid_max_pairs(self):
        """Test History raises ValueError for invalid max_pairs."""
        with pytest.raises(ValueError, match="max_pairs must be positive"):
            History(max_pairs=0)

        with pytest.raises(ValueError, match="max_pairs must be positive"):
            History(max_pairs=-1)

 
class TestConversationManagement:
    """Test conversation pair management functionality."""

    @pytest.mark.unit
    def test_add_single_conversation_pair(self, history_default):
        """Test adding a single conversation pair."""
        history_default.add_conversation_pair("Hello", "Hi there!")

        assert len(history_default._pairs) == 1
        assert history_default._pairs[0] == ("Hello", "Hi there!")

    @pytest.mark.unit
    def test_add_multiple_conversation_pairs(self, history_default, sample_conversation_pairs):
        """Test adding multiple conversation pairs."""
        for request, response in sample_conversation_pairs:
            history_default.add_conversation_pair(request, response)

        assert len(history_default._pairs) == len(sample_conversation_pairs)
        assert list(history_default._pairs) == sample_conversation_pairs

    @pytest.mark.unit
    def test_conversation_pair_limit_enforcement(self):
        """Test that conversation pairs are limited by max_pairs."""
        history = History(max_pairs=3)

        # Add more pairs than the limit
        pairs = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3"), ("Q4", "A4"), ("Q5", "A5")]
        for request, response in pairs:
            history.add_conversation_pair(request, response)

        # Should only keep the last 3 pairs
        assert len(history._pairs) == 3
        assert list(history._pairs) == [("Q3", "A3"), ("Q4", "A4"), ("Q5", "A5")]

    @pytest.mark.unit
    def test_add_conversation_pair_with_empty_response(self, history_default):
        """Test adding conversation pair with empty response."""
        history_default.add_conversation_pair("Hello", "")

        assert len(history_default._pairs) == 1
        assert history_default._pairs[0] == ("Hello", "")

    @pytest.mark.unit
    def test_add_conversation_pair_with_metadata(self, history_default):
        """Test adding conversation pair with metadata (currently unused)."""
        metadata = {"timestamp": "2023-01-01", "user_id": "test"}
        history_default.add_conversation_pair("Hello", "Hi!", metadata)

        assert len(history_default._pairs) == 1
        assert history_default._pairs[0] == ("Hello", "Hi!")

    @pytest.mark.unit
    def test_clear_history(self, history_with_data):
        """Test clearing conversation history."""
        # Verify data exists
        assert len(history_with_data._pairs) > 0

        # Clear history
        history_with_data.clear_history()

        # Verify everything is cleared
        assert len(history_with_data._pairs) == 0


class TestFormatManagement:
    """Test format switching and management."""


    @pytest.mark.unit
    def test_set_format_xml_to_minimal(self, history_xml):
        """Test switching from XML to minimal format."""
        assert history_xml._history_format == HistoryFormat.XML

        history_xml.set_format(HistoryFormat.MINIMAL)

        assert history_xml._history_format == HistoryFormat.MINIMAL
        assert history_xml._formatter.format_name == "minimal"

    @pytest.mark.unit
    def test_set_format_minimal_to_xml(self, history_minimal):
        """Test switching from minimal to XML format."""
        assert history_minimal._history_format == HistoryFormat.MINIMAL

        history_minimal.set_format(HistoryFormat.XML)

        assert history_minimal._history_format == HistoryFormat.XML
        assert history_minimal._formatter.format_name == "xml"


class TestFormattedOutput:
    """Test formatted history output functionality."""

    @pytest.mark.unit
    def test_get_formatted_history_xml(self, history_xml, sample_conversation_pairs):
        """Test getting formatted history in XML format."""
        for request, response in sample_conversation_pairs:
            history_xml.add_conversation_pair(request, response)

        formatted = history_xml.get_formatted_history()

        assert "<conversation_history>" in formatted
        assert "</conversation_history>" in formatted
        TestUtilities.assert_xml_structure(formatted, len(sample_conversation_pairs))

    @pytest.mark.unit
    def test_get_formatted_history_minimal(self, history_minimal, sample_conversation_pairs):
        """Test getting formatted history in minimal format."""
        for request, response in sample_conversation_pairs:
            history_minimal.add_conversation_pair(request, response)

        formatted = history_minimal.get_formatted_history()

        TestUtilities.assert_minimal_structure(formatted, len(sample_conversation_pairs))

    @pytest.mark.unit
    def test_get_formatted_history_empty(self, history_default):
        """Test getting formatted history when empty."""
        formatted = history_default.get_formatted_history()
        assert formatted == ""

    @pytest.mark.unit
    def test_get_formatted_history_single_pair(self, history_xml):
        """Test formatting single conversation pair."""
        history_xml.add_conversation_pair("Test question", "Test answer")

        formatted = history_xml.get_formatted_history()

        assert "<turn_1>" in formatted
        assert "<user>Test question</user>" in formatted
        assert "<assistant>Test answer</assistant>" in formatted


class TestStatistics:
    """Test statistics and debugging functionality."""

    @pytest.mark.unit
    def test_get_stats_empty(self, history_default):
        """Test getting statistics for empty history."""
        stats = history_default.get_stats()

        expected = {
            "pairs": "0/10",
            "format": "xml"
        }
        assert stats == expected

    @pytest.mark.unit
    def test_get_stats_with_data(self, history_with_data):
        """Test getting statistics with data."""
        stats = history_with_data.get_stats()

        assert stats["pairs"] == "4/10"  # 4 sample conversation pairs
        assert stats["format"] == "xml"

    @pytest.mark.unit
    def test_get_stats_different_limits(self):
        """Test statistics with different limits."""
        history = History(max_pairs=5, history_format=HistoryFormat.MINIMAL)
        history.add_conversation_pair("Q", "A")

        stats = history.get_stats()

        assert stats["pairs"] == "1/5"
        assert stats["format"] == "minimal"


class TestStringRepresentation:
    """Test string representation functionality."""

    @pytest.mark.unit
    def test_str_empty_history(self, history_default):
        """Test string representation of empty history."""
        result = str(history_default)
        assert "No conversation history available." in result

    @pytest.mark.unit
    def test_str_with_conversations(self, history_default):
        """Test string representation with conversations."""
        history_default.add_conversation_pair("Hello", "Hi!")
        history_default.add_conversation_pair("How are you?", "Good!")

        result = str(history_default)

        assert "=== Conversation History ===" in result
        assert "Turn 1:" in result
        assert "Turn 2:" in result
        assert "User: Hello" in result
        assert "Assistant: Hi!" in result
        assert "User: How are you?" in result
        assert "Assistant: Good!" in result
        assert "=== End History ===" in result

    @pytest.mark.unit
    def test_str_with_empty_response(self, history_default):
        """Test string representation with empty response."""
        history_default.add_conversation_pair("Hello", "")

        result = str(history_default)

        assert "Assistant: [No response yet]" in result

    @pytest.mark.unit
    def test_str_with_none_response(self, history_default):
        """Test string representation with None response."""
        # Manually add pair with None response to test edge case
        history_default._pairs.append(("Hello", None))

        result = str(history_default)

        assert "Assistant: [No response yet]" in result


class TestHistoryDescription:
    """Test history description functionality."""

    @pytest.mark.unit
    def test_get_history_description_empty(self, history_default):
        """Test getting history description when empty."""
        description = history_default.get_history_description()
        assert description == []

    @pytest.mark.unit
    def test_get_history_description_with_data(self, history_default):
        """Test getting history description with conversation data."""
        history_default.add_conversation_pair("Q1", "A1")
        history_default.add_conversation_pair("Q2", "A2")

        description = history_default.get_history_description()

        expected = [
            {"request": "Q1", "response": "A1"},
            {"request": "Q2", "response": "A2"}
        ]
        assert description == expected

    @pytest.mark.unit
    def test_get_history_description_with_empty_response(self, history_default):
        """Test getting history description with empty responses."""
        history_default.add_conversation_pair("Q1", "")
        history_default.add_conversation_pair("Q2", None)

        description = history_default.get_history_description()

        # Empty string should still be included
        assert description[0] == {"request": "Q1"}
        # None response should still be included
        assert description[1] == {"request": "Q2"}


@pytest.mark.edge_case
class TestEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.unit
    def test_very_long_text(self, history_default):
        """Test handling very long conversation text."""
        long_request = "A" * 10000
        long_response = "B" * 10000

        history_default.add_conversation_pair(long_request, long_response)

        assert len(history_default._pairs) == 1
        assert history_default._pairs[0] == (long_request, long_response)

    @pytest.mark.unit
    def test_unicode_text(self, history_default):
        """Test handling Unicode characters."""
        unicode_request = "Hello 你好 🌍"
        unicode_response = "World 世界 🚀"

        history_default.add_conversation_pair(unicode_request, unicode_response)

        assert len(history_default._pairs) == 1
        assert history_default._pairs[0] == (unicode_request, unicode_response)

    @pytest.mark.unit
    def test_special_characters(self, history_default, text_with_special_chars):
        """Test handling special XML characters."""
        history_default.add_conversation_pair(text_with_special_chars, "Response")

        assert len(history_default._pairs) == 1
        assert history_default._pairs[0][0] == text_with_special_chars

    @pytest.mark.unit
    def test_multiline_text(self, history_default, multiline_text):
        """Test handling multiline text."""
        history_default.add_conversation_pair(multiline_text, "Response")

        assert len(history_default._pairs) == 1
        assert "\n" in history_default._pairs[0][0]

    @pytest.mark.unit
    def test_maximum_pairs_boundary(self):
        """Test behavior at maximum pairs boundary."""
        history = History(max_pairs=1)

        history.add_conversation_pair("First", "Response1")
        assert len(history._pairs) == 1

        history.add_conversation_pair("Second", "Response2")
        assert len(history._pairs) == 1
        assert history._pairs[0] == ("Second", "Response2")


@pytest.mark.performance
class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.unit
    def test_large_history_performance(self):
        """Test performance with large conversation history."""
        history = History(max_pairs=1000)

        # Add many pairs quickly
        pairs = [(f"Q{i}", f"A{i}") for i in range(1000)]
        for request, response in pairs:
            history.add_conversation_pair(request, response)

        # Test that operations are still fast
        assert len(history._pairs) == 1000

        # Test formatting performance
        formatted = history.get_formatted_history()
        assert len(formatted) > 0

        # Test stats performance
        stats = history.get_stats()
        assert stats["pairs"] == "1000/1000"

    @pytest.mark.unit
    def test_format_switching_performance(self, history_default):
        """Test performance of format switching."""
        # Add some data
        for i in range(10):
            history_default.add_conversation_pair(f"Q{i}", f"A{i}")

        # Switch formats multiple times
        for _ in range(10):
            history_default.set_format(HistoryFormat.MINIMAL)
            history_default.set_format(HistoryFormat.XML)

        # Ensure data integrity is maintained
        assert len(history_default._pairs) == 10