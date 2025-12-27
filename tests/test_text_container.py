"""Test TEXT context items with TextContainer and diagnostic task."""
import pytest
from src.vlmchat.pipeline.task_base import Context, ContextDataType
from src.vlmchat.pipeline.text_container import TextContainer
from src.vlmchat.pipeline.cached_item_types import CachedItemType
from src.vlmchat.pipeline.diagnostic_task import DiagnosticTask


def test_text_container_formats():
    """Test TextContainer format conversions."""
    # Create text container with string
    text = "Hello, World!\nThis is a test."
    container = TextContainer(text, cache_key="test_text_1")
    
    # Test str format (should be cached)
    assert container.has_format("str")
    assert container.get("str") == text
    
    # Test bytes format (lazy conversion)
    assert not container.has_format("bytes")
    bytes_data = container.get("bytes")
    assert isinstance(bytes_data, bytes)
    assert bytes_data == text.encode("utf-8")
    assert container.has_format("bytes")
    
    # Test lines format (lazy conversion)
    assert not container.has_format("lines")
    lines_data = container.get("lines")
    assert isinstance(lines_data, list)
    assert lines_data == ["Hello, World!", "This is a test."]
    assert container.has_format("lines")
    
    # Verify all formats are now cached
    cached = container.get_cached_formats()
    assert "str" in cached
    assert "bytes" in cached
    assert "lines" in cached


def test_text_container_bytes_roundtrip():
    """Test converting bytes back to string."""
    # Create from string
    text = "Test message"
    container = TextContainer(text, cache_key="test_text_2")
    
    # Convert to bytes
    bytes_data = container.get("bytes")
    
    # Clear str cache to force reverse conversion
    container._cached_formats.pop("str")
    
    # Convert back to string
    recovered = container.get("str")
    assert recovered == text


def test_text_container_lines_roundtrip():
    """Test converting lines back to string."""
    # Create from string
    text = "Line 1\nLine 2\nLine 3"
    container = TextContainer(text, cache_key="test_text_3")
    
    # Convert to lines
    lines = container.get("lines")
    assert lines == ["Line 1", "Line 2", "Line 3"]
    
    # Clear str cache
    container._cached_formats.pop("str")
    
    # Convert back to string
    recovered = container.get("str")
    assert recovered == text


def test_text_container_metadata():
    """Test TextContainer metadata support."""
    metadata = {"source": "test", "timestamp": 123456}
    container = TextContainer("test", cache_key="test_text_4", metadata=metadata)
    
    assert container.metadata == metadata


def test_text_in_context():
    """Test using TextContainer in Context with default label."""
    context = Context()
    
    # Add text to context using default label
    text1 = TextContainer("First message", cache_key="msg1")
    text2 = TextContainer("Second message", cache_key="msg2")
    
    # Initialize with default label
    context.data[ContextDataType.TEXT] = {"default": [text1, text2]}
    
    # Verify structure
    assert ContextDataType.TEXT in context.data
    assert "default" in context.data[ContextDataType.TEXT]
    assert len(context.data[ContextDataType.TEXT]["default"]) == 2
    
    # Retrieve and verify
    items = context.data[ContextDataType.TEXT]["default"]
    assert items[0].get("str") == "First message"
    assert items[1].get("str") == "Second message"


def test_text_with_labels():
    """Test using TextContainer with multiple labels."""
    context = Context()
    
    # Add texts with different labels
    log_text = TextContainer("Log entry 1", cache_key="log1")
    output_text = TextContainer("Output result", cache_key="out1")
    
    context.data[ContextDataType.TEXT] = {
        "log": [log_text],
        "output": [output_text]
    }
    
    # Verify label separation
    assert "log" in context.data[ContextDataType.TEXT]
    assert "output" in context.data[ContextDataType.TEXT]
    assert context.data[ContextDataType.TEXT]["log"][0].get("str") == "Log entry 1"
    assert context.data[ContextDataType.TEXT]["output"][0].get("str") == "Output result"


def test_diagnostic_with_text_context():
    """Test diagnostic task with TEXT context items."""
    # Create context with text
    context = Context()
    text1 = TextContainer("Diagnostic test message", cache_key="diag1")
    context.data[ContextDataType.TEXT] = {"default": [text1]}
    
    # Run diagnostic task
    task = DiagnosticTask(message="text_test", task_id="diag1")
    result = task.run(context)
    
    # Verify text preserved
    assert ContextDataType.TEXT in result.data
    assert "default" in result.data[ContextDataType.TEXT]
    assert len(result.data[ContextDataType.TEXT]["default"]) == 1
    assert result.data[ContextDataType.TEXT]["default"][0].get("str") == "Diagnostic test message"


def test_text_container_unsupported_format():
    """Test TextContainer with unsupported format."""
    container = TextContainer("test", cache_key="test_text_5")
    
    with pytest.raises(ValueError, match="Unsupported text format"):
        container.get("invalid_format")


def test_text_container_multiple_conversions():
    """Test multiple format conversions maintain consistency."""
    text = "Multi-format test\nWith newlines"
    container = TextContainer(text, cache_key="test_text_6")
    
    # Convert to multiple formats
    str1 = container.get("str")
    bytes1 = container.get("bytes")
    lines1 = container.get("lines")
    
    # Get them again (should use cache)
    str2 = container.get("str")
    bytes2 = container.get("bytes")
    lines2 = container.get("lines")
    
    # Verify consistency
    assert str1 == str2
    assert bytes1 == bytes2
    assert lines1 == lines2
    assert str1 == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
