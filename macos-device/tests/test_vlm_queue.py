"""Tests for VLMQueue routing logic."""

import pytest
from camera_framework import Buffer, drop_oldest_policy, blocking_policy
from macos_device.vlm_queue import VLMQueue


class MockSmolVLMWorker:
    """Mock SmolVLM worker for testing."""
    
    def __init__(self, busy: bool = False):
        self._busy = busy
    
    def is_busy(self) -> bool:
        return self._busy
    
    def set_busy(self, busy: bool) -> None:
        self._busy = busy


def create_confirmation_event(track_id: int, vlm_required: bool = False, vlm_reasoning: str = "") -> dict:
    """Helper to create confirmation event."""
    return {
        "event_type": "confirmation",
        "track_id": track_id,
        "vlm_required": vlm_required,
        "vlm_reasoning": vlm_reasoning,
        "timestamp": 1234567890.0,
    }


def test_vlm_queue_routes_to_alerts_when_not_required():
    """VLMQueue should route to alerts when vlm_required=False."""
    # Setup
    mock_worker = MockSmolVLMWorker(busy=False)
    vlm_queue = VLMQueue(max_queue_size=10, smolvlm_worker=mock_worker)
    
    # Create buffers
    input_buffer = Buffer(size=5, policy=drop_oldest_policy)
    alert_buffer = Buffer(size=5, policy=drop_oldest_policy)
    vlm_buffer = Buffer(size=1, policy=blocking_policy)
    
    # Wire up
    vlm_queue.add_input("in", input_buffer)
    vlm_queue.add_output("alerts", alert_buffer)
    vlm_queue.add_output("vlm", vlm_buffer)
    
    # Send confirmation that doesn't require VLM
    message = {}
    event = create_confirmation_event(track_id=1, vlm_required=False)
    message.setdefault("confirmations", []).append(event)
    input_buffer.put(message)
    
    # Process
    vlm_queue.process()
    
    # Should route to alerts, not VLM
    assert alert_buffer.has_data()
    assert not vlm_buffer.has_data()
    
    # Verify event in alerts
    alert_message = alert_buffer.get()
    alerts = alert_message.get("confirmations", [])
    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 1
    
    # Check stats
    stats = vlm_queue.get_stats()
    assert stats['routed_to_alerts'] == 1
    assert stats['routed_to_vlm'] == 0


def test_vlm_queue_routes_to_vlm_when_ready():
    """VLMQueue should route to VLM when vlm_required=True and worker ready."""
    # Setup
    mock_worker = MockSmolVLMWorker(busy=False)  # Ready
    vlm_queue = VLMQueue(max_queue_size=10, smolvlm_worker=mock_worker)
    
    # Create buffers
    input_buffer = Buffer(size=5, policy=drop_oldest_policy)
    alert_buffer = Buffer(size=5, policy=drop_oldest_policy)
    vlm_buffer = Buffer(size=1, policy=blocking_policy)
    
    # Wire up
    vlm_queue.add_input("in", input_buffer)
    vlm_queue.add_output("alerts", alert_buffer)
    vlm_queue.add_output("vlm", vlm_buffer)
    
    # Send confirmation that requires VLM
    message = {}
    event = create_confirmation_event(
        track_id=1,
        vlm_required=True,
        vlm_reasoning="Verify person is wearing blue shirt"
    )
    message.setdefault("confirmations", []).append(event)
    input_buffer.put(message)
    
    # Process
    vlm_queue.process()
    
    # Should route to VLM, not alerts
    assert vlm_buffer.has_data()
    assert not alert_buffer.has_data()
    
    # Verify event in VLM buffer
    vlm_message = vlm_buffer.get()
    vlm_events = vlm_message.get("confirmations", [])
    assert len(vlm_events) == 1
    assert vlm_events[0]["track_id"] == 1
    assert vlm_events[0]["vlm_reasoning"] == "Verify person is wearing blue shirt"
    
    # Check stats
    stats = vlm_queue.get_stats()
    assert stats['routed_to_alerts'] == 0
    assert stats['routed_to_vlm'] == 1


def test_vlm_queue_queues_when_busy():
    """VLMQueue should queue messages when VLM worker is busy."""
    # Setup
    mock_worker = MockSmolVLMWorker(busy=True)  # Busy
    vlm_queue = VLMQueue(max_queue_size=10, smolvlm_worker=mock_worker)
    
    # Create buffers
    input_buffer = Buffer(size=5, policy=drop_oldest_policy)
    alert_buffer = Buffer(size=5, policy=drop_oldest_policy)
    vlm_buffer = Buffer(size=1, policy=blocking_policy)
    
    # Wire up
    vlm_queue.add_input("in", input_buffer)
    vlm_queue.add_output("alerts", alert_buffer)
    vlm_queue.add_output("vlm", vlm_buffer)
    
    # Send confirmation that requires VLM
    message = {}
    event = create_confirmation_event(track_id=1, vlm_required=True)
    message.setdefault("confirmations", []).append(event)
    input_buffer.put(message)
    
    # Process
    vlm_queue.process()
    
    # Should queue (not route to either output)
    assert not alert_buffer.has_data()
    assert not vlm_buffer.has_data()
    
    # Check queue
    stats = vlm_queue.get_stats()
    assert stats['queue_size'] == 1
    assert stats['queued'] == 1
    
    # Now mark worker as ready
    mock_worker.set_busy(False)
    
    # Process again (should dequeue)
    vlm_queue.process()
    
    # Should now be in VLM buffer
    assert vlm_buffer.has_data()
    assert not alert_buffer.has_data()
    
    # Queue should be empty
    stats = vlm_queue.get_stats()
    assert stats['queue_size'] == 0
    assert stats['routed_to_vlm'] == 1


def test_vlm_queue_ejects_oldest_when_full():
    """VLMQueue should eject oldest to alerts with timeout when queue full."""
    # Setup
    mock_worker = MockSmolVLMWorker(busy=True)  # Busy
    vlm_queue = VLMQueue(max_queue_size=2, smolvlm_worker=mock_worker)
    
    # Create buffers
    input_buffer = Buffer(size=10, policy=drop_oldest_policy)
    alert_buffer = Buffer(size=10, policy=drop_oldest_policy)
    vlm_buffer = Buffer(size=1, policy=blocking_policy)
    
    # Wire up
    vlm_queue.add_input("in", input_buffer)
    vlm_queue.add_output("alerts", alert_buffer)
    vlm_queue.add_output("vlm", vlm_buffer)
    
    # Send 3 confirmations (queue size is 2)
    for i in range(3):
        message = {}
        event = create_confirmation_event(track_id=i+1, vlm_required=True)
        message.setdefault("confirmations", []).append(event)
        input_buffer.put(message)
        vlm_queue.process()
    
    # First 2 should be queued, 3rd should eject oldest
    stats = vlm_queue.get_stats()
    assert stats['queue_size'] == 2
    assert stats['queued'] == 3
    assert stats['timeouts'] == 1
    assert stats['queue_full_drops'] == 1
    
    # Should have ejected track 1 to alerts with timeout
    assert alert_buffer.has_data()
    alert_message = alert_buffer.get()
    alerts = alert_message.get("confirmations", [])
    assert len(alerts) == 1
    assert alerts[0]["track_id"] == 1
    assert alerts[0]["vlm_timeout"] is True
    assert alerts[0]["vlm_verified"] is False


def test_vlm_queue_handles_multiple_confirmations():
    """VLMQueue should process multiple confirmations in one context."""
    # Setup
    mock_worker = MockSmolVLMWorker(busy=False)
    vlm_queue = VLMQueue(max_queue_size=10, smolvlm_worker=mock_worker)
    
    # Create buffers
    input_buffer = Buffer(size=5, policy=drop_oldest_policy)
    alert_buffer = Buffer(size=5, policy=drop_oldest_policy)
    vlm_buffer = Buffer(size=5, policy=drop_oldest_policy)
    
    # Wire up
    vlm_queue.add_input("in", input_buffer)
    vlm_queue.add_output("alerts", alert_buffer)
    vlm_queue.add_output("vlm", vlm_buffer)
    
    # Send context with multiple confirmations
    message = {}
    message.setdefault("confirmations", []).append(create_confirmation_event(1, vlm_required=False))
    message.setdefault("confirmations", []).append(create_confirmation_event(2, vlm_required=True))
    message.setdefault("confirmations", []).append(create_confirmation_event(3, vlm_required=False))
    input_buffer.put(message)
    
    # Process
    vlm_queue.process()
    
    # Should route 2 to alerts, 1 to VLM
    assert alert_buffer.has_data()
    assert vlm_buffer.has_data()
    
    stats = vlm_queue.get_stats()
    assert stats['routed_to_alerts'] == 2
    assert stats['routed_to_vlm'] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
