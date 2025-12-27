"""
Pytest configuration and fixtures for prompt module tests.

This module provides shared fixtures, test utilities, and configuration
for testing the prompt management components.
"""

import pytest
from PIL import Image
from collections import deque
from typing import List, Tuple
import io

from src.prompt.history import History
from src.prompt.history_format import HistoryFormat
from src.prompt.prompt import Prompt
from src.prompt.history_format_xml import HistoryFormatXML
from src.prompt.history_format_minimal import HistoryFormatMinimal
from src.prompt.history_format_factory import HistoryFormatFactory


@pytest.fixture
def sample_image():
    """Create a test PIL image."""
    # Create a simple 100x100 RGB image
    image = Image.new('RGB', (100, 100), color='red')
    return image


@pytest.fixture
def sample_conversation_pairs() -> List[Tuple[str, str]]:
    """Sample conversation pairs for testing."""
    return [
        ("Hello", "Hi there!"),
        ("How are you?", "I'm doing well, thank you!"),
        ("What can you see?", "I can see a beautiful image."),
        ("Tell me more", "It appears to be a landscape scene.")
    ]


@pytest.fixture
def long_conversation_pairs() -> List[Tuple[str, str]]:
    """Long conversation pairs exceeding typical limits."""
    return [
        (f"Question {i}", f"Answer {i}") for i in range(1, 16)
    ]


@pytest.fixture
def conversation_deque(sample_conversation_pairs) -> deque:
    """Convert conversation pairs to deque format."""
    return deque(sample_conversation_pairs)


@pytest.fixture
def empty_deque() -> deque:
    """Empty conversation deque."""
    return deque()


@pytest.fixture
def xml_formatter():
    """XML formatter instance."""
    return HistoryFormatXML()


@pytest.fixture
def minimal_formatter():
    """Minimal formatter instance."""
    return HistoryFormatMinimal()


@pytest.fixture
def minimal_formatter_custom_limit():
    """Minimal formatter with custom word limit."""
    return HistoryFormatMinimal(word_limit=5)


@pytest.fixture
def history_default():
    """Default History instance."""
    return History()


@pytest.fixture
def history_xml():
    """History instance with XML formatting."""
    return History(
        max_pairs=10,
        max_images=1,
        history_format=HistoryFormat.XML
    )


@pytest.fixture
def history_minimal():
    """History instance with minimal formatting."""
    return History(
        max_pairs=5,
        max_images=1,
        history_format=HistoryFormat.MINIMAL
    )


@pytest.fixture
def history_with_data(history_xml, sample_conversation_pairs, sample_image):
    """History instance pre-populated with test data."""
    for request, response in sample_conversation_pairs:
        history_xml.add_conversation_pair(request, response)
    return history_xml


@pytest.fixture
def prompt_default():
    """Default Prompt instance."""
    return Prompt()


@pytest.fixture
def prompt_minimal():
    """Prompt instance with minimal formatting."""
    return Prompt(
        max_pairs=5,
        history_format=HistoryFormat.MINIMAL
    )


@pytest.fixture
def prompt_with_data(prompt_default, sample_conversation_pairs, sample_image):
    """Prompt instance pre-populated with test data."""
    for request, response in sample_conversation_pairs:
        prompt_default.add_assistant_response(request, response)
    prompt_default.current_image = sample_image
    return prompt_default


@pytest.fixture
def long_text():
    """Long text for truncation testing."""
    return "This is a very long piece of text that should be truncated by the minimal formatter when the word limit is exceeded and it continues with even more words."


@pytest.fixture
def short_text():
    """Short text that should not be truncated."""
    return "Short text"


@pytest.fixture
def empty_text():
    """Empty text for edge case testing."""
    return ""


@pytest.fixture
def text_with_special_chars():
    """Text containing special characters."""
    return "Text with <special> &characters; and \"quotes\" and 'apostrophes'"


@pytest.fixture
def multiline_text():
    """Text containing line breaks."""
    return "Line one\nLine two\nLine three"


class TestUtilities:
    """Utility functions for testing."""

    @staticmethod
    def create_pairs_deque(pairs: List[Tuple[str, str]]) -> deque:
        """Convert list of pairs to deque."""
        return deque(pairs)

    @staticmethod
    def create_test_image(width: int = 50, height: int = 50, color: str = 'blue') -> Image.Image:
        """Create a test image with specified dimensions and color."""
        return Image.new('RGB', (width, height), color=color)

    @staticmethod
    def count_words(text: str) -> int:
        """Count words in text."""
        return len(text.split())

    @staticmethod
    def assert_xml_structure(xml_text: str, expected_turns: int):
        """Assert XML text has expected structure."""
        assert "<conversation_history>" in xml_text
        assert "</conversation_history>" in xml_text
        for i in range(1, expected_turns + 1):
            assert f"<turn_{i}>" in xml_text
            assert f"</turn_{i}>" in xml_text

    @staticmethod
    def assert_minimal_structure(minimal_text: str, expected_turns: int):
        """Assert minimal text has expected structure."""
        lines = minimal_text.strip().split('\n')
        assert len(lines) == expected_turns
        for line in lines:
            assert "U:" in line
            if " | A:" in line:
                assert line.count("U:") == 1
                assert line.count("A:") == 1


# Custom pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "performance: mark test as a performance test")
    config.addinivalue_line("markers", "edge_case: mark test as an edge case test")