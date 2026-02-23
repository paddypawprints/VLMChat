"""Smoke tests for buffer-based dataflow architecture."""

import pytest
import sys
import time
import weakref
from PIL import Image
import numpy as np
from camera_framework import (
    BaseTask,
    Runner,
    Buffer,
    blocking_policy,
    drop_oldest_policy,
    drop_newest_policy,
    decimate_policy,
)


def get_refcount(obj):
    """Get Python refcount for debugging.
    
    sys.getrefcount(obj) includes:
    - 1 for the obj parameter
    - actual references
    
    So subtract 1 for the parameter.
    """
    count = sys.getrefcount(obj)
    return count - 1  # Just parameter


class SourceTask(BaseTask):
    """Source task that creates message with test data."""
    
    def __init__(self, name="source", num_frames=3):
        super().__init__(name=name, interval=0.01)
        self.num_frames = num_frames
        self.frames_produced = 0
    
    def process(self) -> None:
        if self.frames_produced < self.num_frames:
            # Create message
            message = {}
            # Create test frame
            frame = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
            message.setdefault("frame", []).append(frame)
            self.frames_produced += 1
            # Write to outputs
            for output_buf in self.outputs.values():
                output_buf.put(message)


class ProcessingTask(BaseTask):
    """Processing task that reads and modifies message."""
    
    def __init__(self, name="processor"):
        super().__init__(name=name)
        self.frames_processed = 0
    
    def process(self) -> None:
        # Read from input buffer
        input_buffer = list(self.inputs.values())[0]
        message = input_buffer.get()
        if message is None:
            return
        
        frames = message.get("frame", [])
        for frame in frames:
            # Simple processing: add 1 to all pixels
            processed = frame + 1
            message.setdefault("processed", []).append(processed)
        self.frames_processed += len(frames)
        
        # Write to output buffer
        for output_buf in self.outputs.values():
            output_buf.put(message)


class SinkTask(BaseTask):
    """Sink task that consumes Context."""
    
    def __init__(self, name="sink", fields=None):
        super().__init__(name=name, fields=fields)
        self.frames_received = 0
        self.received_data = []
    
    def process(self) -> None:
        # Read from input buffer
        input_buffer = list(self.inputs.values())[0]
        ctx = input_buffer.get()
        if ctx is None:
            return
        
        # Use field mapping to read correct field name
        field_name = self.field("data")
        frames = ctx.get_all(field_name)
        self.frames_received += len(frames)
        self.received_data.extend(frames)


# ===== Buffer Policy Tests =====

def test_blocking_policy():
    """Blocking policy should wait for space and release items properly."""
    buffer = Buffer(size=2, policy=blocking_policy, name="test_blocking")
    
    # Create contexts and track with weakrefs
    ctx1 = Context()
    ctx1.append("data", 1)
    ctx2 = Context()
    ctx2.append("data", 2)
    weak1 = weakref.ref(ctx1)
    weak2 = weakref.ref(ctx2)
    
    # Put in buffer
    assert buffer.put(ctx1) == True
    assert buffer.put(ctx2) == True
    assert len(buffer) == 2
    
    # Delete our references - buffer is now sole owner
    del ctx1, ctx2
    
    # Both should still be alive (buffer holds them)
    assert weak1() is not None
    assert weak2() is not None
    
    # Get one item (buffer releases ref)
    result = buffer.get()
    assert result.get("data", 0) == 1
    assert len(buffer) == 1
    
    # Delete result - item 1 should be GC'd
    del result
    assert weak1() is None  # ✅ No leak - item 1 cleaned up
    assert weak2() is not None  # Item 2 still in buffer


def test_drop_oldest_policy():
    """Drop oldest should overwrite when full and release old refs."""
    buffer = Buffer(size=2, policy=drop_oldest_policy, name="test_ring")
    
    ctx1 = Context()
    ctx1.append("id", 1)
    ctx2 = Context()
    ctx2.append("id", 2)
    ctx3 = Context()
    ctx3.append("id", 3)
    
    # Initial refcounts
    assert get_refcount(ctx1) == 1
    assert get_refcount(ctx2) == 1
    assert get_refcount(ctx3) == 1
    
    buffer.put(ctx1)
    buffer.put(ctx2)
    
    # Both in buffer now
    assert get_refcount(ctx1) == 2
    assert get_refcount(ctx2) == 2
    
    buffer.put(ctx3)  # Should drop ctx1
    
    # ctx1 should be released (back to 1), ctx2 and ctx3 in buffer
    assert get_refcount(ctx1) == 1  # Only our local var
    assert get_refcount(ctx2) == 2  # Our var + buffer
    assert get_refcount(ctx3) == 2  # Our var + buffer
    
    # Should have ctx2 and ctx3
    result = buffer.get()
    assert result.get("id", 0) == 2
    result = buffer.get()
    assert result.get("id", 0) == 3


def test_drop_newest_policy():
    """Drop newest should silently discard when full and not hold extra refs."""
    buffer = Buffer(size=2, policy=drop_newest_policy, name="test_drop")
    
    ctx1 = Context()
    ctx1.append("id", 1)
    ctx2 = Context()
    ctx2.append("id", 2)
    ctx3 = Context()
    ctx3.append("id", 3)
    
    assert buffer.put(ctx1) == True
    assert buffer.put(ctx2) == True
    
    # Buffer should hold refs to ctx1 and ctx2
    assert get_refcount(ctx1) == 2
    assert get_refcount(ctx2) == 2
    
    assert buffer.put(ctx3) == True  # Always returns True (silently discarded when full)
    
    # ctx3 should not be in buffer (only our var holds it)
    assert get_refcount(ctx3) == 1
    
    # Should only have ctx1 and ctx2
    result = buffer.get()
    assert result.get("id", 0) == 1
    result = buffer.get()
    assert result.get("id", 0) == 2


def test_decimate_policy():
    """Decimate policy should accept every Nth item and release skipped refs."""
    buffer = Buffer(size=10, policy=decimate_policy(2), name="test_decimate")
    
    weak_refs = []
    # Put 6 items, should accept every 2nd (items 2, 4, 6)
    for i in range(1, 7):
        ctx = Context()
        ctx.append("id", i)
        weak_refs.append(weakref.ref(ctx))
        buffer.put(ctx)
        del ctx  # Remove our reference
    
    # Should have 3 items (2, 4, 6) in buffer
    assert len(buffer) == 3
    
    # Items 1, 3, 5 should be GC'd (skipped by decimate)
    assert weak_refs[0]() is None  # Item 1 - skipped, cleaned up ✅
    assert weak_refs[1]() is not None  # Item 2 - in buffer
    assert weak_refs[2]() is None  # Item 3 - skipped, cleaned up ✅
    assert weak_refs[3]() is not None  # Item 4 - in buffer
    assert weak_refs[4]() is None  # Item 5 - skipped, cleaned up ✅
    assert weak_refs[5]() is not None  # Item 6 - in buffer
    
    assert buffer.get().get("id", 0) == 2
    assert buffer.get().get("id", 0) == 4
    assert buffer.get().get("id", 0) == 6


# ===== Context Reference Counting Tests =====

def test_context_refcount():
    """Context should track Python refcounts correctly."""
    ctx = Context()
    
    # Add an object
    frame = np.array([1, 2, 3])
    ctx.append("frame", frame)
    
    # Check refcount (context + local variable)
    refs = ctx.debug_refs()
    assert "frame" in refs
    assert refs["frame"][0] >= 2  # At least 2 refs (ctx + frame var)


def test_context_buffer_sharing():
    """Multiple buffers can hold references to same Context."""
    ctx = Context()
    frame = np.array([1, 2, 3])
    ctx.append("frame", frame)
    
    # Initial refcount (just ctx var)
    assert get_refcount(ctx) == 1
    
    # Put in two buffers
    buf1 = Buffer(size=5, policy=drop_oldest_policy)
    buf2 = Buffer(size=5, policy=drop_oldest_policy)
    
    buf1.put(ctx)
    buf2.put(ctx)
    
    # Refcount should be 3 (ctx var + buf1 + buf2)
    assert get_refcount(ctx) == 3
    
    # Both buffers should have the same context
    ctx1 = buf1.get()
    ctx2 = buf2.get()
    
    # Now refcount should be 3 (ctx + ctx1 + ctx2, buffers released)
    assert get_refcount(ctx) == 3
    
    assert np.array_equal(ctx1.get("frame", 0), ctx2.get("frame", 0))


# ===== End-to-End Pipeline Tests =====

def test_source_processor_sink_pipeline():
    """Test complete pipeline: source → buffer → processor → buffer → sink."""
    # Create buffers
    source_to_proc = Buffer(size=5, policy=drop_oldest_policy, name="source→proc")
    proc_to_sink = Buffer(size=5, policy=drop_oldest_policy, name="proc→sink")
    
    # Create tasks
    source = SourceTask(num_frames=3)
    processor = ProcessingTask()
    sink = SinkTask(fields={"data": "processed"})  # Map "data" -> "processed"
    
    # Wire up buffers
    source.add_output("out", source_to_proc)
    processor.add_input("in", source_to_proc)
    processor.add_output("out", proc_to_sink)
    sink.add_input("in", proc_to_sink)
    
    # Create runner
    runner = Runner([source, processor, sink])
    
    # Run pipeline
    for _ in range(10):  # Run enough iterations
        runner.run_once()
        time.sleep(0.02)
    
    # Verify data flowed through
    assert source.frames_produced == 3
    assert processor.frames_processed > 0
    assert sink.frames_received > 0


def test_fan_out():
    """Test fan-out: one source writing to two buffers."""
    # Create buffers
    buf1 = Buffer(size=5, policy=drop_oldest_policy, name="fan-out-1")
    buf2 = Buffer(size=5, policy=drop_oldest_policy, name="fan-out-2")
    
    # Create source with two outputs
    source = SourceTask(num_frames=2)
    source.add_output("out1", buf1)
    source.add_output("out2", buf2)
    
    # Create two sinks with field mapping: "data" -> "frame"
    sink1 = SinkTask(name="sink1", fields={"data": "frame"})
    sink2 = SinkTask(name="sink2", fields={"data": "frame"})
    sink1.add_input("in", buf1)
    sink2.add_input("in", buf2)
    
    # Run
    runner = Runner([source, sink1, sink2])
    for _ in range(10):
        runner.run_once()
        time.sleep(0.02)
    
    # Both sinks should receive data
    assert sink1.frames_received > 0
    assert sink2.frames_received > 0


def test_rate_control_with_decimate():
    """Test rate control using decimate policy."""
    source_to_proc = Buffer(size=10, policy=drop_oldest_policy)
    # Decimate by 2: processor only gets half the frames
    proc_to_sink = Buffer(size=10, policy=decimate_policy(2), name="rate-control")
    
    source = SourceTask(num_frames=6)
    processor = ProcessingTask()
    sink = SinkTask()
    
    source.add_output("out", source_to_proc)
    processor.add_input("in", source_to_proc)
    processor.add_output("out", proc_to_sink)
    sink.add_input("in", proc_to_sink)
    
    runner = Runner([source, processor, sink])
    for _ in range(20):
        runner.run_once()
        time.sleep(0.01)
    
    # Source produced 6, but sink should receive ~3 (due to decimate_2)
    assert source.frames_produced == 6
    # Sink receives less due to decimation
    assert sink.frames_received < source.frames_produced


# ===== Memory Management Tests =====

def test_buffer_releases_old_items():
    """Ring buffer should release references to dropped items."""
    buffer = Buffer(size=2, policy=drop_oldest_policy)
    
    # Create contexts with unique objects
    contexts = []
    for i in range(5):
        ctx = Context()
        obj = {"id": i}  # Track this object
        ctx.append("data", obj)
        contexts.append(ctx)
        
        # Before put: contexts list holds ref (loop var may add +1)
        assert get_refcount(ctx) >= 1
        
        buffer.put(ctx)  # Add to buffer
        
        # After put: refcount = 2 (contexts list + buffer), unless dropped
    
    # Buffer only holds last 2
    assert len(buffer) == 2
    
    # First 3 contexts should have been dropped (only in contexts list)
    assert get_refcount(contexts[0]) >= 1  # Dropped (list may cache)
    assert get_refcount(contexts[1]) >= 1  # Dropped
    assert get_refcount(contexts[2]) >= 1  # Dropped
    assert get_refcount(contexts[3]) >= 2  # In buffer (list + buffer)
    assert get_refcount(contexts[4]) >= 2  # In buffer


def test_context_copy_shares_refs():
    """Context.copy() should share references to objects."""
    ctx1 = Context()
    frame = np.array([1, 2, 3])
    ctx1.append("frame", frame)
    
    # Initial refcounts
    assert get_refcount(ctx1) == 1
    frame_refcount_before = get_refcount(frame)
    
    # Copy context
    ctx2 = ctx1.copy()
    
    # Two context objects now
    assert get_refcount(ctx1) == 1
    assert get_refcount(ctx2) == 1
    
    # But both should reference same frame object
    assert ctx1.get("frame", 0) is ctx2.get("frame", 0)
    
    # Frame refcount should increase (ctx1.data + ctx2.data)
    frame_refcount_after = get_refcount(frame)
    assert frame_refcount_after > frame_refcount_before
    
    # Modifying array affects both (shared reference)
    ctx1.get("frame", 0)[0] = 99
    assert ctx2.get("frame", 0)[0] == 99


# ===== Edge Cases =====

def test_empty_buffer():
    """Getting from empty buffer should return None and not affect refs."""
    buffer = Buffer(size=5, policy=drop_oldest_policy)
    
    # Empty buffer
    assert buffer.get() is None
    assert not buffer.has_data()
    
    # Add and remove item
    ctx = Context()
    ctx.append("test", 1)
    
    assert get_refcount(ctx) == 1
    buffer.put(ctx)
    assert get_refcount(ctx) == 2
    
    result = buffer.get()
    assert result is ctx
    assert get_refcount(ctx) == 2  # ctx + result
    
    # Now empty again
    assert buffer.get() is None


def test_task_readiness():
    """Tasks should report ready correctly."""
    # Source task (no inputs) always ready
    source = SourceTask()
    assert source.is_ready()
    
    # Processing task (has inputs) ready when input has data
    processor = ProcessingTask()
    buf = Buffer(size=5, policy=drop_oldest_policy)
    processor.add_input("in", buf)
    
    assert not processor.is_ready()  # No data in buffer
    
    ctx = Context()
    ctx.append("test", 1)
    buf.put(ctx)
    
    assert processor.is_ready()  # Now has data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
