"""
Smoke tests for Jetson buffer pool implementation.

Tests buffer acquisition, release, auto-promotion, and pool pressure.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from vlmchat.pipeline.sources.jetson_camera import BufferPool, PooledBuffer


def test_buffer_pool_acquire_release():
    """Test basic buffer acquisition and release."""
    print("\n=== Test: Buffer Pool Acquire/Release ===")
    
    pool = BufferPool(num_buffers=5, width=100, height=100)
    
    # Acquire buffer
    buf = pool.acquire()
    assert buf is not None, "Should acquire buffer from pool"
    assert buf.refcount == 1, "Buffer should have refcount=1 after acquire"
    assert buf.valid is True, "Buffer should be valid"
    
    # Release buffer
    pool.release(buf)
    assert buf.refcount == 0, "Buffer should have refcount=0 after release"
    
    print("✓ Acquire/release works correctly")


def test_buffer_pool_exhaustion():
    """Test pool behavior when exhausted."""
    print("\n=== Test: Pool Exhaustion ===")
    
    pool = BufferPool(num_buffers=3, width=100, height=100)
    
    # Acquire all buffers
    buffers = []
    for i in range(3):
        buf = pool.acquire()
        assert buf is not None, f"Should acquire buffer {i}"
        buffers.append(buf)
    
    # Check pressure
    pressure = pool.get_pressure()
    assert pressure == 1.0, f"Pressure should be 1.0 (100%), got {pressure}"
    
    # Try to acquire when pool full - should promote and reclaim oldest
    buf4 = pool.acquire()
    assert buf4 is not None, "Should acquire buffer after reclaiming oldest"
    # buf4 should be the recycled oldest buffer (buffers[0])
    assert buf4.index == buffers[0].index, "Should reuse oldest buffer"
    assert buf4.valid is True, "Recycled buffer should be valid again"
    
    metrics = pool.get_metrics()
    assert metrics['buffers_evicted'] == 1, "Should have evicted one buffer"
    
    print("✓ Pool exhaustion handled correctly")


def test_buffer_pool_metrics():
    """Test pool metrics tracking."""
    print("\n=== Test: Pool Metrics ===")
    
    pool = BufferPool(num_buffers=5, width=100, height=100)
    
    # Acquire and release
    buf1 = pool.acquire()
    buf2 = pool.acquire()
    pool.release(buf1)
    
    metrics = pool.get_metrics()
    
    assert metrics['total_buffers'] == 5
    assert metrics['borrowed_buffers'] == 1  # buf2 still borrowed
    assert metrics['free_buffers'] == 4
    assert metrics['pressure'] == 0.2  # 1/5
    assert metrics['buffers_acquired'] == 2
    assert metrics['buffers_released'] == 1
    
    print(f"✓ Metrics: {metrics}")


def test_buffer_data_isolation():
    """Test that different buffers have independent data."""
    print("\n=== Test: Buffer Data Isolation ===")
    
    pool = BufferPool(num_buffers=3, width=10, height=10)
    
    # Acquire two buffers
    buf1 = pool.acquire()
    buf2 = pool.acquire()
    
    # Write different data
    buf1.data[:] = 100
    buf2.data[:] = 200
    
    # Verify isolation
    assert np.all(buf1.data == 100), "buf1 should contain 100"
    assert np.all(buf2.data == 200), "buf2 should contain 200"
    
    pool.release(buf1)
    pool.release(buf2)
    
    print("✓ Buffer data is isolated")


def test_buffer_reuse():
    """Test that released buffers can be reused."""
    print("\n=== Test: Buffer Reuse ===")
    
    pool = BufferPool(num_buffers=2, width=10, height=10)
    
    # First cycle
    buf1 = pool.acquire()
    buf1_id = buf1.index
    pool.release(buf1)
    
    # Second cycle - should reuse same buffer
    buf2 = pool.acquire()
    assert buf2.index == buf1_id, "Should reuse released buffer"
    
    pool.release(buf2)
    
    print("✓ Buffers are reused after release")


def test_concurrent_references():
    """Test buffer with multiple container references."""
    print("\n=== Test: Concurrent References ===")
    
    pool = BufferPool(num_buffers=3, width=10, height=10)
    
    # Mock container class
    class MockContainer:
        def __init__(self, name):
            self.name = name
    
    buf = pool.acquire()
    
    # Add multiple references
    cont1 = MockContainer("cont1")
    cont2 = MockContainer("cont2")
    
    buf.add_ref(cont1)
    buf.add_ref(cont2)
    
    assert buf.refcount == 3, "Should have 3 refs (1 camera + 2 containers)"
    assert len(buf.containers) == 2, "Should track 2 containers"
    
    # Remove references
    buf.remove_ref(cont1)
    assert buf.refcount == 2, "Should have 2 refs after removing one"
    
    buf.remove_ref(cont2)
    pool.release(buf)
    assert buf.refcount == 0, "Should have 0 refs after all removed"
    
    print("✓ Multiple references handled correctly")


def test_promotion_tracking():
    """Test that promotion metrics are tracked."""
    print("\n=== Test: Promotion Tracking ===")
    
    pool = BufferPool(num_buffers=2, width=10, height=10)
    
    # Acquire both buffers
    buf1 = pool.acquire()
    buf2 = pool.acquire()
    
    # Add mock container to buf1
    class MockContainer:
        pass
    
    cont = MockContainer()
    buf1.add_ref(cont)
    
    # Try to acquire third - should evict buf1
    buf3 = pool.acquire()
    
    metrics = pool.get_metrics()
    assert metrics['buffers_evicted'] == 1, "Should track eviction"
    # frames_promoted only increments for containers with _promote_to_owned method
    # MockContainer doesn't have it, so count stays 0
    assert metrics['frames_promoted'] == 0, "Mock containers don't promote"
    
    print(f"✓ Eviction tracked: {metrics}")


def run_all_tests():
    """Run all smoke tests."""
    print("=" * 60)
    print("JETSON BUFFER POOL SMOKE TESTS")
    print("=" * 60)
    
    tests = [
        test_buffer_pool_acquire_release,
        test_buffer_pool_exhaustion,
        test_buffer_pool_metrics,
        test_buffer_data_isolation,
        test_buffer_reuse,
        test_concurrent_references,
        test_promotion_tracking,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {failed} TESTS FAILED")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
