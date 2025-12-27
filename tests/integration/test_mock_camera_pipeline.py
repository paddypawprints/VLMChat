#!/usr/bin/env python3
"""
Integration tests for mock camera pipeline with buffer pool.

Simulates a camera source generating frames into a pool and tasks
consuming them, validating zero-copy operation, pressure management,
and automatic promotion under realistic conditions.
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vlmchat.pipeline.cache.image import ImageContainer
from vlmchat.pipeline.sources.jetson_camera import BufferPool
from vlmchat.pipeline.image.formats import ImageFormat


class MockCamera:
    """Mock camera that generates frames into a buffer pool."""
    
    def __init__(self, pool: BufferPool):
        self.pool = pool
        self.frame_count = 0
    
    def capture_frame(self) -> ImageContainer:
        """Capture a frame into pooled buffer."""
        # Acquire buffer from pool
        buffer = self.pool.acquire()
        
        # Generate test pattern (frame number encoded)
        buffer.data[:] = self.frame_count % 256
        self.pool.release(buffer)  # Release camera ref
        
        # Create pooled container
        container = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
        
        self.frame_count += 1
        return container


class MockTask:
    """Mock task that processes frames."""
    
    def __init__(self, name: str, processing_time: float = 0.0):
        self.name = name
        self.processing_time = processing_time
        self.frames_processed = 0
    
    def process(self, container: ImageContainer) -> None:
        """Process a frame (read data)."""
        # Read data (zero-copy if pooled)
        data = container.get(ImageFormat.NUMPY)
        
        # Simulate processing
        if self.processing_time > 0:
            time.sleep(self.processing_time)
        
        # Verify data readable
        _ = data[0, 0, 0]
        
        self.frames_processed += 1


def test_mock_camera_basic_capture():
    """Test basic frame capture from mock camera."""
    print("\n=== Test: Mock Camera Basic Capture ===")
    
    # Create pool and camera
    pool = BufferPool(num_buffers=10, width=640, height=480, channels=3)
    camera = MockCamera(pool)
    
    # Capture frame
    container = camera.capture_frame()
    
    # Verify container is pooled
    assert container.is_pooled(), "Container should be pooled"
    
    # Verify data accessible
    data = container.get(ImageFormat.NUMPY)
    assert data.shape == (480, 640, 3), "Frame shape correct"
    assert (data == 0).all(), "First frame should be 0"
    
    # Capture second frame
    container2 = camera.capture_frame()
    data2 = container2.get(ImageFormat.NUMPY)
    assert (data2 == 1).all(), "Second frame should be 1"
    
    print("✓ Mock camera captures frames correctly")


def test_mock_camera_pipeline_flow():
    """Test camera → task pipeline flow."""
    print("\n=== Test: Camera → Task Pipeline Flow ===")
    
    # Create pool, camera, and task
    pool = BufferPool(num_buffers=10, width=100, height=100, channels=3)
    camera = MockCamera(pool)
    task = MockTask("detector")
    
    # Process multiple frames
    num_frames = 5
    containers = []
    
    for _ in range(num_frames):
        # Camera captures frame
        container = camera.capture_frame()
        containers.append(container)
        
        # Task processes frame
        task.process(container)
    
    # Verify task processed all frames
    assert task.frames_processed == num_frames, "All frames processed"
    
    # Verify all containers still valid
    for i, container in enumerate(containers):
        data = container.get(ImageFormat.NUMPY)
        assert (data == i).all(), f"Frame {i} data preserved"
    
    print("✓ Pipeline flow works correctly")


def test_mock_camera_pool_pressure():
    """Test pool pressure under continuous capture."""
    print("\n=== Test: Pool Pressure Under Capture ===")
    
    # Create small pool
    pool = BufferPool(num_buffers=5, width=100, height=100, channels=3)
    camera = MockCamera(pool)
    
    # Track pressure as we capture
    containers = []
    pressures = []
    
    # Capture frames (hold references)
    for _ in range(5):
        container = camera.capture_frame()
        containers.append(container)
        pressures.append(pool.get_pressure())
    
    # Verify increasing pressure
    assert pressures[0] == 0.2, "20% after 1 frame"
    assert pressures[4] == 1.0, "100% when full"
    
    # All containers should be pooled
    for container in containers:
        assert container.is_pooled(), "All should be pooled"
    
    # Capture one more (triggers promotion)
    container6 = camera.capture_frame()
    
    # First container should be promoted
    assert not containers[0].is_pooled(), "Oldest promoted"
    assert container6.is_pooled(), "New frame pooled"
    
    print("✓ Pool pressure managed correctly")


def test_mock_camera_rapid_capture_release():
    """Test rapid capture and release - verify system handles it gracefully."""
    print("\n=== Test: Rapid Capture and Release ===")
    
    # Create pool
    pool = BufferPool(num_buffers=5, width=100, height=100, channels=3)
    camera = MockCamera(pool)
    task = MockTask("fast_processor")
    
    # Capture and release many frames
    num_frames = 50
    
    for _ in range(num_frames):
        # Capture frame
        container = camera.capture_frame()
        
        # Process immediately
        task.process(container)
        
        # Release (goes out of scope)
        del container
    
    # Verify system handled it (didn't crash, processed all frames)
    assert task.frames_processed == num_frames, "All frames should be processed"
    
    # Verify pool is still functional
    test_buffer = pool.acquire()
    assert test_buffer is not None, "Pool should still be functional"
    
    print("✓ Rapid capture/release handled gracefully")


def test_mock_camera_slow_consumer():
    """Test behavior with slow consumer (accumulation)."""
    print("\n=== Test: Slow Consumer Accumulation ===")
    
    # Create small pool
    pool = BufferPool(num_buffers=3, width=100, height=100, channels=3)
    camera = MockCamera(pool)
    
    # Capture frames faster than processing (accumulate)
    containers = []
    
    # Capture 5 frames (pool only has 3)
    for i in range(5):
        container = camera.capture_frame()
        containers.append(container)
    
    # Verify promotions occurred
    metrics = pool.get_metrics()
    assert metrics["frames_promoted"] == 2, "Should promote 2 frames"
    
    # Verify first 2 promoted, last 3 pooled
    assert not containers[0].is_pooled(), "Oldest promoted"
    assert not containers[1].is_pooled(), "Second oldest promoted"
    assert containers[2].is_pooled(), "Third should be pooled"
    assert containers[3].is_pooled(), "Fourth should be pooled"
    assert containers[4].is_pooled(), "Newest should be pooled"
    
    # Verify all data still accessible
    for i, container in enumerate(containers):
        data = container.get(ImageFormat.NUMPY)
        assert (data == i).all(), f"Frame {i} data preserved"
    
    print("✓ Slow consumer handled with promotion")


def test_mock_camera_multiple_tasks():
    """Test multiple tasks sharing frames."""
    print("\n=== Test: Multiple Tasks Sharing Frames ===")
    
    # Create pool and camera
    pool = BufferPool(num_buffers=10, width=100, height=100, channels=3)
    camera = MockCamera(pool)
    
    # Create multiple tasks
    detector = MockTask("detector")
    classifier = MockTask("classifier")
    tracker = MockTask("tracker")
    
    # Capture frame
    container = camera.capture_frame()
    
    # Multiple tasks process same frame
    detector.process(container)
    classifier.process(container)
    tracker.process(container)
    
    # Verify all tasks processed
    assert detector.frames_processed == 1, "Detector processed"
    assert classifier.frames_processed == 1, "Classifier processed"
    assert tracker.frames_processed == 1, "Tracker processed"
    
    # Verify container still valid and pooled
    assert container.is_pooled(), "Should still be pooled"
    data = container.get(ImageFormat.NUMPY)
    assert (data == 0).all(), "Data still accessible"
    
    print("✓ Multiple tasks can share frames")


def test_mock_camera_mixed_hold_release():
    """Test mixed pattern of holding and releasing frames."""
    print("\n=== Test: Mixed Hold/Release Pattern ===")
    
    # Create pool
    pool = BufferPool(num_buffers=5, width=100, height=100, channels=3)
    camera = MockCamera(pool)
    
    # Pattern: capture, hold some, release others
    held_containers = []
    
    # Capture 10 frames, hold every other one
    for i in range(10):
        container = camera.capture_frame()
        
        if i % 2 == 0:
            held_containers.append(container)
        # else: released (goes out of scope)
    
    # Verify we held 5 frames
    assert len(held_containers) == 5, "Should hold 5 frames"
    
    # Verify all held frames still valid
    for i, container in enumerate(held_containers):
        data = container.get(ImageFormat.NUMPY)
        expected_value = i * 2  # 0, 2, 4, 6, 8
        assert (data == expected_value).all(), f"Frame {i*2} preserved"
    
    # Some should be promoted (pool only has 5 buffers)
    promoted_count = sum(1 for c in held_containers if not c.is_pooled())
    pooled_count = sum(1 for c in held_containers if c.is_pooled())
    
    # Expect newest frames pooled, oldest promoted
    assert promoted_count > 0, "Some frames should be promoted"
    assert pooled_count > 0, "Some frames should be pooled"
    
    print(f"✓ Mixed pattern handled: {promoted_count} promoted, {pooled_count} pooled")


def test_mock_camera_zero_copy_validation():
    """Test that zero-copy is actually happening."""
    print("\n=== Test: Zero-Copy Validation ===")
    
    # Create pool and camera
    pool = BufferPool(num_buffers=10, width=100, height=100, channels=3)
    camera = MockCamera(pool)
    
    # Capture frame
    container = camera.capture_frame()
    
    # Get data (should be zero-copy)
    data = container.get(ImageFormat.NUMPY)
    
    # Verify it's actually the pool buffer
    # In pooled mode, get() should return pool_buffer.data directly
    assert container.is_pooled(), "Should be pooled"
    
    # Modify through returned array
    data[0, 0, 0] = 255
    
    # Get again - should see modification (same array)
    data2 = container.get(ImageFormat.NUMPY)
    assert data2[0, 0, 0] == 255, "Should see modification (zero-copy)"
    assert data2 is data, "Should be same array object"
    
    print("✓ Zero-copy validated")


def run_all_tests():
    """Run all mock camera pipeline integration tests."""
    tests = [
        test_mock_camera_basic_capture,
        test_mock_camera_pipeline_flow,
        test_mock_camera_pool_pressure,
        test_mock_camera_rapid_capture_release,
        test_mock_camera_slow_consumer,
        test_mock_camera_multiple_tasks,
        test_mock_camera_mixed_hold_release,
        test_mock_camera_zero_copy_validation,
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("MOCK CAMERA PIPELINE INTEGRATION TESTS")
    print("=" * 60)
    
    for test in tests:
        try:
            test()
            passed += 1
            print("✅ PASSED")
        except AssertionError as e:
            failed += 1
            print(f"❌ FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {e}")
    
    print("=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {failed} TESTS FAILED")
    print("=" * 60)
    
    return failed


if __name__ == "__main__":
    sys.exit(run_all_tests())
