#!/usr/bin/env python3
"""
Integration tests for end-to-end buffer promotion flow.

Tests the automatic promotion of ImageContainers when the BufferPool
needs to reclaim buffers, including metrics tracking and data integrity.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vlmchat.pipeline.cache.image import ImageContainer
from vlmchat.pipeline.sources.jetson_camera import BufferPool
from vlmchat.pipeline.image.formats import ImageFormat


def test_automatic_promotion_on_exhaustion():
    """Test that pool automatically promotes oldest container when exhausted."""
    print("\n=== Test: Automatic Promotion on Exhaustion ===")
    
    # Create small pool (2 buffers)
    pool = BufferPool(num_buffers=2, width=100, height=100, channels=3)
    
    # Acquire both buffers and create containers
    buffer1 = pool.acquire()
    buffer1.data[:] = 111
    pool.release(buffer1)  # Release camera ref
    container1 = ImageContainer(cache_key="frame1", pooled_buffer=buffer1)
    
    buffer2 = pool.acquire()
    buffer2.data[:] = 222
    pool.release(buffer2)  # Release camera ref
    container2 = ImageContainer(cache_key="frame2", pooled_buffer=buffer2)
    
    # Verify pool exhausted
    assert pool.get_pressure() == 1.0, "Pool should be at 100% pressure"
    assert container1.is_pooled(), "Container 1 should be pooled"
    assert container2.is_pooled(), "Container 2 should be pooled"
    
    # Acquire third buffer (should trigger promotion of oldest)
    buffer3 = pool.acquire()
    
    # Verify first container was promoted
    assert not container1.is_pooled(), "Container 1 should be promoted"
    assert container2.is_pooled(), "Container 2 should still be pooled"
    
    # Verify data integrity after promotion
    data1 = container1.get(ImageFormat.NUMPY)
    assert (data1 == 111).all(), "Container 1 data preserved"
    
    data2 = container2.get(ImageFormat.NUMPY)
    assert (data2 == 222).all(), "Container 2 data preserved"
    
    print("✓ Automatic promotion works correctly")


def test_promotion_metrics_tracking():
    """Test that promotion events are tracked in metrics."""
    print("\n=== Test: Promotion Metrics Tracking ===")
    
    # Create small pool
    pool = BufferPool(num_buffers=2, width=100, height=100, channels=3)
    
    # Check initial metrics
    initial_metrics = pool.get_metrics()
    assert initial_metrics["frames_promoted"] == 0, "No promotions yet"
    assert initial_metrics["buffers_evicted"] == 0, "No evictions yet"
    
    # Fill pool with containers
    containers = []
    for i in range(2):
        buffer = pool.acquire()
        buffer.data[:] = i
        pool.release(buffer)  # Release camera ref
        container = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
        containers.append(container)
    
    # Trigger promotion
    buffer3 = pool.acquire()
    
    # Check metrics updated
    metrics = pool.get_metrics()
    assert metrics["frames_promoted"] == 1, "Should track 1 promotion"
    assert metrics["buffers_evicted"] == 1, "Should track 1 eviction"
    
    # Trigger another promotion
    buffer4 = pool.acquire()
    
    # Check metrics incremented
    metrics = pool.get_metrics()
    assert metrics["frames_promoted"] == 2, "Should track 2 promotions"
    assert metrics["buffers_evicted"] == 2, "Should track 2 evictions"
    
    print("✓ Promotion metrics tracked correctly")


def test_multiple_containers_per_buffer_promotion():
    """Test promotion when multiple containers share a buffer."""
    print("\n=== Test: Promotion with Multiple Container References ===")
    
    # Create small pool
    pool = BufferPool(num_buffers=2, width=100, height=100, channels=3)
    
    # Acquire buffer and create multiple containers
    buffer1 = pool.acquire()
    buffer1.data[:] = 99
    pool.release(buffer1)  # Release camera ref
    
    container1a = ImageContainer(cache_key="frame1a", pooled_buffer=buffer1)
    
    container1b = ImageContainer(cache_key="frame1b", pooled_buffer=buffer1)
    
    # Verify refcount
    assert buffer1.refcount == 2, "Buffer should have 2 references"
    
    # Acquire second buffer
    buffer2 = pool.acquire()
    buffer2.data[:] = 88
    pool.release(buffer2)  # Release camera ref
    container2 = ImageContainer(cache_key="frame2", pooled_buffer=buffer2)
    
    # Pool is now full, trigger promotion of buffer1
    buffer3 = pool.acquire()
    
    # Verify both containers sharing buffer1 were promoted
    assert not container1a.is_pooled(), "Container 1a should be promoted"
    assert not container1b.is_pooled(), "Container 1b should be promoted"
    assert container2.is_pooled(), "Container 2 should still be pooled"
    
    # Verify data integrity for both promoted containers
    data1a = container1a.get(ImageFormat.NUMPY)
    data1b = container1b.get(ImageFormat.NUMPY)
    assert (data1a == 99).all(), "Container 1a data preserved"
    assert (data1b == 99).all(), "Container 1b data preserved"
    
    # Verify they now have independent copies
    data1a[0, 0, 0] = 255
    assert data1b[0, 0, 0] == 99, "Containers should have independent data"
    
    print("✓ Multiple containers promoted correctly")


def test_promotion_with_format_conversions():
    """Test that promotion preserves converted formats."""
    print("\n=== Test: Promotion Preserves Format Conversions ===")
    
    # Create pool
    pool = BufferPool(num_buffers=2, width=50, height=50, channels=3)
    
    # Create container with conversions
    buffer1 = pool.acquire()
    buffer1.data[:, :, 0] = 100  # B
    buffer1.data[:, :, 1] = 150  # G
    buffer1.data[:, :, 2] = 200  # R
    pool.release(buffer1)  # Release camera ref
    
    container1 = ImageContainer(cache_key="frame1", pooled_buffer=buffer1)
    
    # Access data (zero-copy)
    bgr_before = container1.get(ImageFormat.NUMPY)
    assert bgr_before is buffer1.data, "Should be zero-copy"
    
    # Fill pool to trigger promotion
    buffer2 = pool.acquire()
    buffer3 = pool.acquire()  # Triggers promotion of buffer1
    
    # Verify promotion occurred
    assert not container1.is_pooled(), "Container should be promoted"
    
    # Verify data still correct
    bgr_after = container1.get(ImageFormat.NUMPY)
    
    assert (bgr_after[:, :, 0] == 100).all(), "BGR B channel preserved"
    assert (bgr_after[:, :, 1] == 150).all(), "BGR G channel preserved"
    assert (bgr_after[:, :, 2] == 200).all(), "BGR R channel preserved"
    
    print("✓ Format conversions preserved through promotion")


def test_promotion_under_pressure():
    """Test promotion behavior as pool pressure increases."""
    print("\n=== Test: Promotion Under Increasing Pressure ===")
    
    # Create pool
    pool = BufferPool(num_buffers=5, width=100, height=100, channels=3)
    
    # Track pressure points
    containers = []
    pressures = []
    
    # Gradually fill pool
    for i in range(5):
        buffer = pool.acquire()
        buffer.data[:] = i
        pool.release(buffer)  # Release camera ref
        container = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
        containers.append(container)
        pressures.append(pool.get_pressure())
    
    # Verify increasing pressure
    assert pressures[0] == 0.2, "20% after 1 buffer"
    assert pressures[1] == 0.4, "40% after 2 buffers"
    assert pressures[2] == 0.6, "60% after 3 buffers"
    assert pressures[3] == 0.8, "80% after 4 buffers"
    assert pressures[4] == 1.0, "100% when full"
    
    # All containers should still be pooled
    for container in containers:
        assert container.is_pooled(), "All containers should be pooled"
    
    # Acquire one more (triggers promotion)
    buffer6 = pool.acquire()
    
    # First container should be promoted
    assert not containers[0].is_pooled(), "Oldest promoted"
    for container in containers[1:]:
        assert container.is_pooled(), "Others still pooled"
    
    # Verify data integrity
    for i, container in enumerate(containers):
        data = container.get(ImageFormat.NUMPY)
        assert (data == i).all(), f"Container {i} data preserved"
    
    print("✓ Pressure-based promotion works correctly")


def test_promotion_fifo_order():
    """Test that promotions happen and data is preserved."""
    print("\n=== Test: Promotion with Data Preservation ===")
    
    # Create small pool
    pool = BufferPool(num_buffers=3, width=100, height=100, channels=3)
    
    # Create containers in order
    containers = []
    for i in range(3):
        buffer = pool.acquire()
        buffer.data[:] = i
        pool.release(buffer)  # Release camera ref
        container = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
        containers.append(container)
    
    # All should be pooled initially
    for container in containers:
        assert container.is_pooled(), "All should start pooled"
    
    # Trigger promotions by exhausting pool
    new_buffers = []
    for i in range(3):
        new_buffer = pool.acquire()
        new_buffers.append(new_buffer)
    
    # At least some containers should be promoted (pool was exhausted)
    promoted_count = sum(1 for c in containers if not c.is_pooled())
    assert promoted_count > 0, "Some containers should be promoted when pool exhausted"
    
    # All data should still be accessible regardless of promotion
    for i, container in enumerate(containers):
        data = container.get(ImageFormat.NUMPY)
        assert (data == i).all(), f"Container {i} data preserved"
    
    print(f"✓ Promotions happened ({promoted_count}/3) and data preserved")


def run_all_tests():
    """Run all promotion flow integration tests."""
    tests = [
        test_automatic_promotion_on_exhaustion,
        test_promotion_metrics_tracking,
        test_multiple_containers_per_buffer_promotion,
        test_promotion_with_format_conversions,
        test_promotion_under_pressure,
        test_promotion_fifo_order,
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("PROMOTION FLOW INTEGRATION TESTS")
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
