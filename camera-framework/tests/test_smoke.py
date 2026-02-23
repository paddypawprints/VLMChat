"""Smoke tests for camera-framework core functionality."""

import pytest
import numpy as np
from camera_framework import BaseTask, Runner, Collector, RateInstrument, Buffer, drop_oldest_policy


class MockCamera(BaseTask):
    """Mock camera that produces test data."""
    
    def __init__(self, fields=None):
        super().__init__("MockCamera", fields)
    
    def process(self) -> None:
        message = {}
        frame = np.array([1, 2, 3])
        message[self.field("frame")] = [frame]
        # Write to outputs
        for output_buf in self.outputs.values():
            output_buf.put(message)


class MockProcessor(BaseTask):
    """Mock processor that transforms data."""
    
    def __init__(self, fields=None):
        super().__init__("MockProcessor", fields)
    
    def process(self) -> None:
        # Read from input
        input_buffer = list(self.inputs.values())[0]
        message = input_buffer.get()
        if message is None:
            return
        
        frame = message.get(self.field("frame"), [])[0]
        result = frame * 2
        message[self.field("result")] = [result]
        
        # Write to outputs
        for output_buf in self.outputs.values():
            output_buf.put(message)


def test_basic_pipeline():
    """Pipeline should execute tasks."""
    camera = MockCamera()
    processor = MockProcessor()
    
    # Connect with buffer
    buffer = Buffer(size=5, policy=drop_oldest_policy)
    camera.add_output("out", buffer)
    processor.add_input("in", buffer)
    
    runner = Runner([camera, processor])
    
    # Run pipeline
    runner.run_once()
    
    # Verify tasks ran
    stats = runner.stats()
    assert stats["tasks"] == 2


def test_field_mapping():
    """Tasks should use field mappings correctly."""
    message = {}
    
    # Camera outputs to "image" instead of "frame"
    camera = MockCamera(fields={"frame": "image"})
    camera.outputs = {}  # No actual outputs needed for this test
    
    # Manually call process logic
    frame = np.array([1, 2, 3])
    message[camera.field("frame")] = [frame]
    
    assert "image" in message
    assert "frame" not in message
    
    # Processor reads from "image", writes to "output"
    processor = MockProcessor(fields={"frame": "image", "result": "output"})
    processor.outputs = {}  # No actual outputs needed
    
    # Manually process
    result = message.get(processor.field("frame"), [])[0] * 2
    message[processor.field("result")] = [result]
    
    assert "output" in message
    output = message.get("output", [])[0]
    np.testing.assert_array_equal(output, np.array([2, 4, 6]))


def test_metrics_basic():
    """Metrics should collect data without errors."""
    collector = Collector("test")
    collector.add_instrument(RateInstrument("test.rate"), "test.events")
    
    # Record some data
    for i in range(10):
        collector.record("test.events", 1.0)
    
    # Get stats
    stats = collector.get_stats("test.rate")
    assert stats is not None
    assert stats["count"] == 10
    assert stats["rate"] > 0


def test_message_list_operations():
    """Messages should handle list operations correctly."""
    message = {}
    
    # Append
    message.setdefault("items", []).append("a")
    message.setdefault("items", []).append("b")
    message.setdefault("items", []).append("c")
    
    assert len(message["items"]) == 3
    assert message["items"][0] == "a"
    assert message["items"][-1] == "c"
    
    # Get slice
    items = message["items"][0:2]
    assert items == ["a", "b"]
    
    # Delete
    del message["items"][1]
    assert len(message["items"]) == 2
    assert message["items"] == ["a", "c"]
