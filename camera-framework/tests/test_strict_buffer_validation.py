"""Test strict buffer validation - multiple writers should crash."""

import pytest
from camera_framework.buffer import Buffer, blocking_policy
from camera_framework.task import BaseTask


class DummyTask(BaseTask):
    """Minimal task for testing."""
    
    def process(self):
        pass


def test_single_writer_to_strict_buffer_ok():
    """Single writer to strict buffer should work fine."""
    buf = Buffer(size=5, policy=blocking_policy, strict=True, name="test_strict")
    task1 = DummyTask(name="task1")
    
    # Should not raise
    task1.add_output("default", buf)
    assert len(buf.writers) == 1
    assert buf.writers[0] == task1


def test_multiple_writers_to_non_strict_buffer_ok():
    """Multiple writers to non-strict buffer should work fine."""
    from camera_framework.buffer import drop_oldest_policy
    
    buf = Buffer(size=5, policy=drop_oldest_policy, strict=False, name="test_non_strict")
    task1 = DummyTask(name="task1")
    task2 = DummyTask(name="task2")
    
    # Should not raise
    task1.add_output("default", buf)
    task2.add_output("default", buf)
    
    assert len(buf.writers) == 2
    assert task1 in buf.writers
    assert task2 in buf.writers


def test_multiple_writers_to_strict_buffer_crashes():
    """Multiple writers to strict buffer should crash with clear error."""
    buf = Buffer(size=5, policy=blocking_policy, strict=True, name="test_strict")
    task1 = DummyTask(name="task1")
    task2 = DummyTask(name="task2")
    
    # First writer OK
    task1.add_output("default", buf)
    
    # Second writer should crash
    with pytest.raises(ValueError) as exc_info:
        task2.add_output("default", buf)
    
    error_msg = str(exc_info.value)
    assert "strict mode" in error_msg.lower()
    assert "2 writers" in error_msg.lower()
    assert "task1" in error_msg
    assert "task2" in error_msg
    assert "drop_oldest_policy" in error_msg or "drop_newest_policy" in error_msg


def test_error_message_quality():
    """Error message should be clear and actionable."""
    buf = Buffer(size=5, policy=blocking_policy, strict=True, name="alert_buffer")
    tracker = DummyTask(name="DetectionTracker")
    vlm_worker = DummyTask(name="VLMWorker")
    
    tracker.add_output("alerts", buf)
    
    with pytest.raises(ValueError) as exc_info:
        vlm_worker.add_output("verified_alerts", buf)
    
    error_msg = str(exc_info.value)
    
    # Should mention buffer name
    assert "alert_buffer" in error_msg
    
    # Should list both conflicting tasks
    assert "DetectionTracker" in error_msg
    assert "VLMWorker" in error_msg
    
    # Should suggest alternatives
    assert "drop_oldest_policy" in error_msg or "drop_newest_policy" in error_msg
    
    # Should explain why
    assert "single producer" in error_msg.lower()
