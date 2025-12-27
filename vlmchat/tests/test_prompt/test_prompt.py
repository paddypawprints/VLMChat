"""
Comprehensive unit tests for the Prompt facade class.

This module tests the Prompt class which serves as a facade over
the History class and provides unified interface for prompt operations.
"""

import pytest
from PIL import Image

from src.prompt.prompt import Prompt
from src.prompt.history import History, HistoryFormat
from .conftest import TestUtilities


class TestPromptInitialization:
    """Test Prompt class initialization and configuration."""

    @pytest.mark.unit
    def test_default_initialization(self):
        """Test Prompt initialization with default parameters."""
        prompt = Prompt()

        assert isinstance(prompt.history, History)
        assert prompt.history.max_pairs == 10
        assert prompt.current_image is None
        assert prompt.user_input is None

    @pytest.mark.unit
    def test_custom_initialization(self):
        """Test Prompt initialization with custom parameters."""
        prompt = Prompt(
            max_pairs=5,
            history_format=HistoryFormat.MINIMAL
        )

        assert prompt.history.max_pairs == 5
        assert prompt.history._history_format == HistoryFormat.MINIMAL

    @pytest.mark.unit
    def test_initialization_with_kwargs(self):
        """Test Prompt initialization with additional kwargs."""
        prompt = Prompt(
            max_pairs=3,
            history_format=HistoryFormat.MINIMAL,
            word_limit=8
        )

        assert prompt.history.max_pairs == 3
        assert prompt.history._formatter.word_limit == 8

    @pytest.mark.unit
    def test_initialization_creates_history_instance(self):
        """Test that initialization creates proper History instance."""
        prompt = Prompt(max_pairs=7)

        history = prompt.history
        assert isinstance(history, History)
        assert history.max_pairs == 7

        # Should be the same instance on multiple accesses
        assert prompt.history is history

    @pytest.mark.unit
    def test_initial_state(self, prompt_default):
        """Test initial state of newly created Prompt."""
        assert prompt_default._current_image is None
        assert prompt_default._user_input is None
        assert len(prompt_default.history._pairs) == 0
        assert prompt_default.current_image is None


class TestPromptProperties:
    """Test Prompt class properties and accessors."""

    @pytest.mark.unit
    def test_current_image_property(self, prompt_default, sample_image):
        """Test current_image property getter."""
        # Initially None
        assert prompt_default.current_image is None

        # Set image and test
        prompt_default._current_image = sample_image
        assert prompt_default.current_image == sample_image

    @pytest.mark.unit
    def test_user_input_property(self, prompt_default):
        """Test user_input property getter."""
        # Initially None
        assert prompt_default.user_input is None

        # Set user input and test
        test_input = "What do you see?"
        prompt_default._user_input = test_input
        assert prompt_default.user_input == test_input

    @pytest.mark.unit
    def test_history_property(self, prompt_default):
        """Test history property getter."""
        history = prompt_default.history
        assert isinstance(history, History)

        # Should return same instance
        assert prompt_default.history is history

    @pytest.mark.unit
    def test_property_types(self, prompt_default, sample_image):
        """Test that properties return expected types."""
        # Set values
        prompt_default._current_image = sample_image
        prompt_default._user_input = "test input"

        # Test types
        assert isinstance(prompt_default.current_image, Image.Image) or prompt_default.current_image is None
        assert isinstance(prompt_default.user_input, str) or prompt_default.user_input is None
        assert isinstance(prompt_default.history, History)


class TestPromptConversationManagement:
    """Test conversation management through the Prompt facade."""

    @pytest.mark.unit
    def test_add_assistant_response(self, prompt_default):
        """Test adding assistant response through facade."""
        prompt_default.add_assistant_response("Hello", "Hi there!")

        # Should be added to underlying history
        assert len(prompt_default.history._pairs) == 1
        assert prompt_default.history._pairs[0] == ("Hello", "Hi there!")

    @pytest.mark.unit
    def test_add_multiple_assistant_responses(self, prompt_default, sample_conversation_pairs):
        """Test adding multiple assistant responses."""
        for request, response in sample_conversation_pairs:
            prompt_default.add_assistant_response(request, response)

        # All should be in history
        assert len(prompt_default.history._pairs) == len(sample_conversation_pairs)
        assert list(prompt_default.history._pairs) == sample_conversation_pairs

    @pytest.mark.unit
    def test_add_assistant_response_delegates_to_history(self, prompt_default):
        """Test that add_assistant_response properly delegates to history."""
        request = "Test question"
        response = "Test answer"

        # Use facade method
        prompt_default.add_assistant_response(request, response)

        # Should match direct history access
        direct_pairs = list(prompt_default.history._pairs)
        assert direct_pairs == [(request, response)]

    @pytest.mark.unit
    def test_add_assistant_response_with_empty_strings(self, prompt_default):
        """Test adding response with empty strings."""
        prompt_default.add_assistant_response("", "")

        assert len(prompt_default.history._pairs) == 1
        assert prompt_default.history._pairs[0] == ("", "")

    @pytest.mark.unit
    def test_conversation_limit_through_facade(self, prompt_minimal):
        """Test conversation limit enforcement through facade."""
        # prompt_minimal has max_pairs=5
        for i in range(7):  # Add more than limit
            prompt_minimal.add_assistant_response(f"Q{i}", f"A{i}")

        # Should only keep last 5
        assert len(prompt_minimal.history._pairs) == 5
        pairs = list(prompt_minimal.history._pairs)
        assert pairs == [(f"Q{i}", f"A{i}") for i in range(2, 7)]


class TestPromptStatistics:
    """Test statistics functionality through the facade."""

    @pytest.mark.unit
    def test_get_stats_empty(self, prompt_default):
        """Test getting statistics for empty prompt."""
        stats = prompt_default.get_stats()

        expected = {
            "pairs": "0/10",
            "format": "xml"
        }
        assert stats == expected

    @pytest.mark.unit
    def test_get_stats_with_data(self, prompt_with_data):
        """Test getting statistics with conversation data."""
        stats = prompt_with_data.get_stats()

        assert stats["pairs"] == "4/10"  # 4 sample conversation pairs
        assert stats["format"] == "xml"

    @pytest.mark.unit
    def test_get_stats_delegates_to_history(self, prompt_default):
        """Test that get_stats delegates to history."""
        prompt_default.add_assistant_response("Q", "A")

        facade_stats = prompt_default.get_stats()
        history_stats = prompt_default.history.get_stats()

        assert facade_stats == history_stats

    @pytest.mark.unit
    def test_get_stats_different_configurations(self, prompt_minimal):
        """Test statistics with different prompt configuration."""
        prompt_minimal.add_assistant_response("Q1", "A1")
        prompt_minimal.add_assistant_response("Q2", "A2")

        stats = prompt_minimal.get_stats()

        assert stats["pairs"] == "2/5"  # max_pairs=5
        assert stats["format"] == "minimal"


class TestPromptUserInput:
    """Test user input management through the facade."""

    @pytest.mark.unit
    def test_user_input_property_access(self, prompt_default):
        """Test user input property access."""
        test_input = "What can you tell me about this image?"

        # Set user input
        prompt_default._user_input = test_input

        # Access through property
        assert prompt_default.user_input == test_input

    @pytest.mark.unit
    def test_user_input_none_handling(self, prompt_default):
        """Test user input None handling."""
        # Initially None
        assert prompt_default.user_input is None

        # Set to None explicitly
        prompt_default._user_input = None
        assert prompt_default.user_input is None

    @pytest.mark.unit
    def test_user_input_empty_string(self, prompt_default):
        """Test user input with empty string."""
        prompt_default._user_input = ""
        assert prompt_default.user_input == ""

    @pytest.mark.unit
    def test_user_input_with_special_chars(self, prompt_default, text_with_special_chars):
        """Test user input with special characters."""
        prompt_default._user_input = text_with_special_chars
        assert prompt_default.user_input == text_with_special_chars

    @pytest.mark.unit
    def test_user_input_multiline(self, prompt_default, multiline_text):
        """Test user input with multiline text."""
        prompt_default._user_input = multiline_text
        assert prompt_default.user_input == multiline_text
        assert "\n" in prompt_default.user_input


class TestPromptHistoryIntegration:
    """Test integration between Prompt facade and History."""

    @pytest.mark.unit
    def test_facade_preserves_history_functionality(self, prompt_default, sample_conversation_pairs):
        """Test that facade preserves all history functionality."""
        # Add data through facade
        for request, response in sample_conversation_pairs:
            prompt_default.add_assistant_response(request, response)

        # Direct history access should work
        formatted = prompt_default.history.get_formatted_history()
        assert len(formatted) > 0

        # Clear through history should work
        prompt_default.history.clear_history()
        assert len(prompt_default.history._pairs) == 0

    @pytest.mark.unit
    def test_history_format_switching(self, prompt_default):
        """Test format switching through history affects facade."""
        prompt_default.add_assistant_response("Test", "Response")

        # Switch format through history
        prompt_default.history.set_format(HistoryFormat.MINIMAL)

        # Stats through facade should reflect change
        stats = prompt_default.get_stats()
        assert stats["format"] == "minimal"

    @pytest.mark.unit
    def test_facade_stats_consistency(self, prompt_default, sample_image):
        """Test consistency between facade and history stats."""
        # Add conversation and image
        prompt_default.add_assistant_response("Q", "A")
        prompt_default.history.current_image = sample_image

        # Stats should be consistent
        facade_stats = prompt_default.get_stats()
        history_stats = prompt_default.history.get_stats()

        assert facade_stats == history_stats

    @pytest.mark.unit
    def test_history_methods_accessible(self, prompt_default):
        """Test that all history methods are accessible."""
        history = prompt_default.history

        # Should have all expected methods
        assert hasattr(history, 'add_conversation_pair')
        assert hasattr(history, 'clear_history')
        assert hasattr(history, 'get_stats')
        assert hasattr(history, 'get_formatted_history')
        assert hasattr(history, 'set_format')

        # Methods should be callable
        assert callable(history.add_conversation_pair)
        assert callable(history.clear_history)


class TestPromptFacadePattern:
    """Test that Prompt properly implements the facade pattern."""

    @pytest.mark.unit
    def test_facade_simplifies_interface(self, prompt_default):
        """Test that facade provides simplified interface."""
        # Facade method is simpler than direct history access
        prompt_default.add_assistant_response("Hello", "Hi!")

        # Equivalent to more verbose history call
        expected_pairs = [("Hello", "Hi!")]
        assert list(prompt_default.history._pairs) == expected_pairs

    @pytest.mark.unit
    def test_facade_encapsulates_complexity(self, prompt_default):
        """Test that facade encapsulates underlying complexity."""
        # User doesn't need to know about History class details
        stats = prompt_default.get_stats()

        # But still gets the full functionality
        assert "pairs" in stats
        assert "format" in stats

    @pytest.mark.unit
    def test_facade_maintains_consistency(self, prompt_default, sample_image):
        """Test that facade maintains state consistency."""
        # Actions through facade
        prompt_default.add_assistant_response("Q", "A")

        # State should be consistent across access methods
        assert len(prompt_default.history._pairs) == 1
        stats = prompt_default.get_stats()
        assert stats["pairs"] == "1/10"

    @pytest.mark.unit
    def test_facade_vs_direct_access(self, prompt_default):
        """Test facade methods vs direct history access."""
        # Using facade
        prompt_default.add_assistant_response("Q1", "A1")

        # Using direct access
        prompt_default.history.add_conversation_pair("Q2", "A2")

        # Both should be in history
        pairs = list(prompt_default.history._pairs)
        assert len(pairs) == 2
        assert ("Q1", "A1") in pairs
        assert ("Q2", "A2") in pairs


@pytest.mark.integration
class TestPromptIntegrationScenarios:
    """Test realistic usage scenarios for the Prompt facade."""

    @pytest.mark.unit
    def test_typical_conversation_flow(self, prompt_default, sample_image):
        """Test typical conversation flow through facade."""
        # Start conversation
        prompt_default._user_input = "Hello"
        prompt_default.current_image = sample_image

        # Add first exchange
        prompt_default.add_assistant_response("Hello", "Hi! I can see an image.")

        # Continue conversation
        prompt_default._user_input = "What do you see?"
        prompt_default.add_assistant_response("What do you see?", "I see a red square.")

        # Check final state
        assert len(prompt_default.history._pairs) == 2
        assert prompt_default.user_input == "What do you see?"
        assert prompt_default.current_image == sample_image

        stats = prompt_default.get_stats()
        assert stats["pairs"] == "2/10"

    @pytest.mark.unit
    def test_format_switching_scenario(self, prompt_default):
        """Test format switching during conversation."""
        # Start with XML (default)
        prompt_default.add_assistant_response("Hi", "Hello!")

        # Get formatted output
        xml_output = prompt_default.history.get_formatted_history()
        assert "<conversation_history>" in xml_output

        # Switch to minimal
        prompt_default.history.set_format(HistoryFormat.MINIMAL)

        # Add more conversation
        prompt_default.add_assistant_response("How are you?", "I'm good!")

        # Get new formatted output
        minimal_output = prompt_default.history.get_formatted_history()
        assert "U:" in minimal_output
        assert "A:" in minimal_output
        assert "<conversation_history>" not in minimal_output

    @pytest.mark.unit
    def test_conversation_limit_scenario(self):
        """Test conversation with limit enforcement."""
        prompt = Prompt(max_pairs=3)

        # Fill beyond limit
        conversations = [
            ("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3"),
            ("Q4", "A4"), ("Q5", "A5")
        ]

        for request, response in conversations:
            prompt.add_assistant_response(request, response)

        # Should only keep last 3
        pairs = list(prompt.history._pairs)
        assert len(pairs) == 3
        assert pairs == [("Q3", "A3"), ("Q4", "A4"), ("Q5", "A5")]

        stats = prompt.get_stats()
        assert stats["pairs"] == "3/3"


@pytest.mark.edge_case
class TestPromptEdgeCases:
    """Test edge cases and error conditions for Prompt."""

    @pytest.mark.unit
    def test_none_user_input_handling(self, prompt_default):
        """Test handling None user input gracefully."""
        prompt_default._user_input = None
        assert prompt_default.user_input is None

        # Should not cause errors in other operations
        stats = prompt_default.get_stats()
        assert isinstance(stats, dict)

    @pytest.mark.unit
    def test_very_long_user_input(self, prompt_default):
        """Test handling very long user input."""
        long_input = "A" * 10000
        prompt_default._user_input = long_input

        assert prompt_default.user_input == long_input
        assert len(prompt_default.user_input) == 10000