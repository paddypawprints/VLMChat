"""
Integration tests for prompt module components.

This module tests the interactions between different components
of the prompt system including History, Prompt, formatters, and factory.
"""

import pytest
from collections import deque
from PIL import Image

from src.prompt.prompt import Prompt
from src.prompt.history import History, HistoryFormat
from src.prompt.history_format_factory import HistoryFormatFactory
from src.prompt.history_format_xml import HistoryFormatXML
from src.prompt.history_format_minimal import HistoryFormatMinimal
from .conftest import TestUtilities


@pytest.mark.integration
class TestPromptHistoryIntegration:
    """Test integration between Prompt facade and History."""

    @pytest.mark.unit
    def test_prompt_facade_full_workflow(self, sample_image, sample_conversation_pairs):
        """Test complete workflow through Prompt facade."""
        # Initialize prompt with specific configuration
        prompt = Prompt(
            max_pairs=5,
            history_format=HistoryFormat.XML
        )

        # Set image through history (as would happen in real usage)
        prompt.current_image = sample_image

        # Add conversations through facade
        for request, response in sample_conversation_pairs:
            prompt.add_assistant_response(request, response)

        # Verify state consistency
        assert len(prompt.history._pairs) == len(sample_conversation_pairs)
        assert prompt.current_image == sample_image

        # Get formatted output
        formatted = prompt.history.get_formatted_history()
        assert "<conversation_history>" in formatted

        # Get statistics
        stats = prompt.get_stats()
        assert stats["pairs"] == f"{len(sample_conversation_pairs)}/5"
        assert stats["format"] == "xml"

    @pytest.mark.unit
    def test_conversation_limit_across_components(self, sample_image):
        """Test conversation limit enforcement across all components."""
        prompt = Prompt(max_pairs=3)

        # Add conversations beyond limit
        conversations = [
            ("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3"),
            ("Q4", "A4"), ("Q5", "A5"), ("Q6", "A6")
        ]

        for request, response in conversations:
            prompt.add_assistant_response(request, response)

        # Verify limit enforcement
        assert len(prompt.history._pairs) == 3
        expected_pairs = [("Q4", "A4"), ("Q5", "A5"), ("Q6", "A6")]
        assert list(prompt.history._pairs) == expected_pairs

        # Verify formatted output contains only recent conversations
        formatted = prompt.history.get_formatted_history()
        assert "Q1" not in formatted
        assert "Q4" in formatted and "Q5" in formatted and "Q6" in formatted

        # Verify statistics reflect current state
        stats = prompt.get_stats()
        assert stats["pairs"] == "3/3"

    @pytest.mark.unit
    def test_image_and_conversation_coordination(self, sample_image):
        """Test coordination between image setting and conversation management."""
        prompt = Prompt()

        # Add conversation without image
        prompt.add_assistant_response("Hello", "Hi!")
        stats_no_image = prompt.get_stats()

        # Set image
        prompt.current_image = sample_image
        stats_with_image = prompt.get_stats()

        # Add more conversation with image
        prompt.add_assistant_response("What do you see?", "I see an image.")

        # Verify both are maintained
        assert len(prompt.history._pairs) == 2
        assert prompt.current_image == sample_image

        # Clear history should clear both
        prompt.history.clear_history()
        assert len(prompt.history._pairs) == 0



@pytest.mark.integration
class TestHistoryFormatterIntegration:
    """Test integration between History and formatters."""

    @pytest.mark.unit
    def test_history_with_xml_formatter_integration(self, sample_conversation_pairs, sample_image):
        """Test History with XML formatter end-to-end."""
        history = History(
            max_pairs=10,
            history_format=HistoryFormat.XML
        )

        # Add data
        for request, response in sample_conversation_pairs:
            history.add_conversation_pair(request, response)
 
        # Get formatted output
        formatted = history.get_formatted_history()

        # Verify XML structure
        assert formatted.startswith("<conversation_history>")
        assert formatted.endswith("</conversation_history>")

        # Verify all conversations included
        for request, response in sample_conversation_pairs:
            assert request in formatted
            assert response in formatted

        # Verify proper XML tags
        for i in range(1, len(sample_conversation_pairs) + 1):
            assert f"<turn_{i}>" in formatted
            assert f"</turn_{i}>" in formatted

    @pytest.mark.unit
    def test_history_with_minimal_formatter_integration(self, sample_conversation_pairs, sample_image):
        """Test History with minimal formatter end-to-end."""
        history = History(
            max_pairs=10,
            history_format=HistoryFormat.MINIMAL,
            word_limit=20
        )

        # Add data
        for request, response in sample_conversation_pairs:
            history.add_conversation_pair(request, response)

        # Get formatted output
        formatted = history.get_formatted_history()

        # Verify minimal structure
        lines = formatted.split('\n')
        assert len(lines) == len(sample_conversation_pairs)

        # Verify all conversations included with proper format
        for i, (request, response) in enumerate(sample_conversation_pairs):
            line = lines[i]
            assert f"U: {request}" in line
            assert f"A: {response}" in line
            assert " | " in line

    @pytest.mark.unit
    def test_runtime_format_switching_integration(self, sample_conversation_pairs):
        """Test switching formatters at runtime."""
        history = History(history_format=HistoryFormat.XML)

        # Add conversations in XML mode
        for request, response in sample_conversation_pairs[:2]:
            history.add_conversation_pair(request, response)

        xml_formatted = history.get_formatted_history()
        assert "<conversation_history>" in xml_formatted

        # Switch to minimal format
        history.set_format(HistoryFormat.MINIMAL)

        # Add more conversations in minimal mode
        for request, response in sample_conversation_pairs[2:]:
            history.add_conversation_pair(request, response)

        minimal_formatted = history.get_formatted_history()
        assert "U:" in minimal_formatted
        assert "<conversation_history>" not in minimal_formatted

        # All conversations should be present in minimal format
        assert len(minimal_formatted.split('\n')) == len(sample_conversation_pairs)

    @pytest.mark.unit
    def test_formatter_configuration_persistence(self):
        """Test that formatter configuration persists through operations."""
        history = History(
            history_format=HistoryFormat.MINIMAL,
            word_limit=5  # Very restrictive
        )

        # Add conversation with long text
        long_request = "This is a very long request that should be truncated"
        long_response = "This is a very long response that should also be truncated"
        history.add_conversation_pair(long_request, long_response)

        formatted = history.get_formatted_history()

        # Should be truncated due to word limit
        assert "..." in formatted
        assert len(formatted.split()) < len((long_request + long_response).split())


@pytest.mark.integration
class TestFactoryHistoryIntegration:
    """Test integration between factory and history components."""

    @pytest.mark.unit
    def test_factory_created_formatter_in_history(self):
        """Test using factory-created formatter in History."""
        # Create formatter through factory
        xml_formatter = HistoryFormatFactory.create_formatter(HistoryFormat.XML)
        minimal_formatter = HistoryFormatFactory.create_formatter(HistoryFormat.MINIMAL)

        # Test formatters work with history data
        pairs = deque([("Question", "Answer")])

        xml_result = xml_formatter.format_history(pairs)
        minimal_result = minimal_formatter.format_history(pairs)

        assert xml_result != minimal_result
        assert "Question" in xml_result and "Answer" in xml_result
        assert "Question" in minimal_result and "Answer" in minimal_result

    @pytest.mark.unit
    def test_history_uses_factory_internally(self):
        """Test that History uses factory to create formatters."""
        # Create history with different formats
        xml_history = History(history_format=HistoryFormat.XML)
        minimal_history = History(history_format=HistoryFormat.MINIMAL)

        # Add same data to both
        xml_history.add_conversation_pair("Test", "Response")
        minimal_history.add_conversation_pair("Test", "Response")

        # Should use correct formatters created by factory
        xml_formatted = xml_history.get_formatted_history()
        minimal_formatted = minimal_history.get_formatted_history()

        assert xml_formatted != minimal_formatted
        assert "<conversation_history>" in xml_formatted
        assert "U:" in minimal_formatted

    @pytest.mark.unit
    def test_factory_formatter_configuration_in_history(self):
        """Test factory formatter configuration works in History."""
        history = History(
            history_format=HistoryFormat.MINIMAL,
            word_limit=3  # Custom configuration
        )

        # Add text that should be truncated
        history.add_conversation_pair("One two three four five", "Six seven eight nine ten")

        formatted = history.get_formatted_history()

        # Should be truncated by custom word limit
        assert "..." in formatted
        parts = formatted.split(" | ")
        user_part = parts[0]  # "U: One two three..."
        assistant_part = parts[1]  # "A: Six seven eight..."

        # Should have exactly 3 words plus "..." for each part
        user_words = user_part.replace("U: ", "").split()
        assert len(user_words) == 4  # 3 words + "..."
        assert user_words[-1] == "..."


@pytest.mark.integration
class TestEndToEndScenarios:
    """Test complete end-to-end scenarios."""

    @pytest.mark.unit
    def test_complete_chat_session_simulation(self, sample_image):
        """Simulate a complete chat session with multiple interactions."""
        # Initialize chat application components
        prompt = Prompt(
            max_pairs=3,  # Limited for testing
            history_format=HistoryFormat.XML
        )

        # User loads image
        prompt.current_image = sample_image

        # Chat session
        chat_exchanges = [
            ("Hello", "Hello! I can see you've loaded an image."),
            ("What do you see in this image?", "I can see a red square in the image."),
            ("Tell me more details", "The image appears to be 100x100 pixels in size."),
            ("What color is dominant?", "Red is the dominant color in this image."),
        ]

        # Process each exchange
        for i, (user_msg, assistant_msg) in enumerate(chat_exchanges):
            prompt._user_input = user_msg
            prompt.add_assistant_response(user_msg, assistant_msg)

        # Verify final state
        assert len(prompt.history._pairs) == 3  # Limited by max_pairs
        assert prompt.current_image == sample_image
        assert prompt.user_input == "What color is dominant?"

        # Verify conversation history contains recent exchanges
        pairs = list(prompt.history._pairs)
        expected_recent = chat_exchanges[-3:]  # Last 3 exchanges
        assert pairs == expected_recent

        # Verify formatted output
        formatted = prompt.history.get_formatted_history()
        assert "<conversation_history>" in formatted
        assert "What do you see" in formatted  # From recent history
        assert "Hello" not in formatted  # Evicted due to limit

        # Verify statistics
        stats = prompt.get_stats()
        assert stats["pairs"] == "3/3"
        assert stats["format"] == "xml"

    @pytest.mark.unit
    def test_format_switching_during_conversation(self, sample_image):
        """Test switching formats during an ongoing conversation."""
        prompt = Prompt(max_pairs=5, history_format=HistoryFormat.XML)

        # Start conversation in XML format
        prompt.add_assistant_response("Hi", "Hello there!")
        prompt.add_assistant_response("What's this?", "It's an image.")

        # Get XML formatted output
        xml_output = prompt.history.get_formatted_history()
        assert "<conversation_history>" in xml_output
        assert "<turn_1>" in xml_output

        # Switch to minimal format mid-conversation
        prompt.history.set_format(HistoryFormat.MINIMAL)

        # Continue conversation
        prompt.add_assistant_response("Nice image", "Thank you!")

        # Get minimal formatted output (should include all conversations)
        minimal_output = prompt.history.get_formatted_history()
        lines = minimal_output.split('\n')
        assert len(lines) == 3  # All three exchanges
        assert all("U:" in line for line in lines)
        assert all(" | A:" in line for line in lines)

        # Original XML format should not appear
        assert "<conversation_history>" not in minimal_output

    @pytest.mark.unit
    def test_conversation_limit_with_format_switching(self):
        """Test conversation limits work correctly with format switching."""
        prompt = Prompt(max_pairs=2, history_format=HistoryFormat.XML)

        # Fill to limit in XML format
        prompt.add_assistant_response("Q1", "A1")
        prompt.add_assistant_response("Q2", "A2")

        # Switch format
        prompt.history.set_format(HistoryFormat.MINIMAL)

        # Add more (should evict oldest)
        prompt.add_assistant_response("Q3", "A3")

        # Verify limit enforcement
        pairs = list(prompt.history._pairs)
        assert len(pairs) == 2
        assert pairs == [("Q2", "A2"), ("Q3", "A3")]

        # Verify minimal format output
        formatted = prompt.history.get_formatted_history()
        assert "U: Q2 | A: A2" in formatted
        assert "U: Q3 | A: A3" in formatted
        assert "Q1" not in formatted

    @pytest.mark.unit
    def test_error_recovery_scenario(self, sample_image):
        """Test system behavior during error recovery scenarios."""
        prompt = Prompt(history_format=HistoryFormat.XML)

        # Normal operation
        prompt.add_assistant_response("Hello", "Hi!")

        # Simulate error condition - try invalid format
        try:
            # This should raise an error if we had an invalid format
            # But our current implementation handles this gracefully
            prompt.history.set_format(HistoryFormat.MINIMAL)
            prompt.add_assistant_response("Recovery test", "System recovered")
        except Exception:
            pytest.fail("System should handle format switching gracefully")

        # Verify system is still functional
        stats = prompt.get_stats()
        assert stats["pairs"] == "2/10"
        assert stats["format"] == "minimal"

        formatted = prompt.history.get_formatted_history()
        assert "Recovery test" in formatted


@pytest.mark.integration
@pytest.mark.performance
class TestIntegrationPerformance:
    """Test performance of integrated components."""

    @pytest.mark.unit
    def test_large_conversation_with_format_switching(self):
        """Test performance with large conversations and format switching."""
        prompt = Prompt(max_pairs=100)

        # Add many conversations
        for i in range(50):
            prompt.add_assistant_response(f"Question {i}", f"Answer {i}")

        # Switch formats multiple times
        for _ in range(5):
            prompt.history.set_format(HistoryFormat.MINIMAL)
            formatted_minimal = prompt.history.get_formatted_history()
            assert len(formatted_minimal) > 0

            prompt.history.set_format(HistoryFormat.XML)
            formatted_xml = prompt.history.get_formatted_history()
            assert len(formatted_xml) > 0

        # Verify final state
        assert len(prompt.history._pairs) == 50
        stats = prompt.get_stats()
        assert stats["pairs"] == "50/100"

    @pytest.mark.unit
    def test_formatter_efficiency_comparison(self):
        """Test and compare efficiency of different formatters."""
        # Create identical conversations for comparison
        conversations = [(f"Q{i}", f"A{i}") for i in range(20)]

        xml_prompt = Prompt(max_pairs=30, history_format=HistoryFormat.XML)
        minimal_prompt = Prompt(max_pairs=30, history_format=HistoryFormat.MINIMAL)

        # Add same conversations to both
        for request, response in conversations:
            xml_prompt.add_assistant_response(request, response)
            minimal_prompt.add_assistant_response(request, response)

        # Get formatted outputs
        xml_formatted = xml_prompt.history.get_formatted_history()
        minimal_formatted = minimal_prompt.history.get_formatted_history()

        # Minimal should be more compact
        assert len(minimal_formatted) < len(xml_formatted)

        # Both should contain same information
        for request, response in conversations:
            assert request in xml_formatted
            assert response in xml_formatted
            assert request in minimal_formatted
            assert response in minimal_formatted