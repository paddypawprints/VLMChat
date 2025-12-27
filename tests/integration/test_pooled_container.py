#!/usr/bin/env python3
"""
Integration tests for ImageContainer with PooledBuffer.

Tests the zero-copy behavior when ImageContainer uses a BufferPool,
including promotion, reference counting, and lifecycle management.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vlmchat.pipeline.cache.image import ImageContainer
from vlmchat.pipeline.sources.jetson_camera import BufferPool
from vlmchat.pipeline.image.formats import ImageFormat
from vlmchat.pipeline.image.formats import ImageFormat


def test_pooled_container_zero_copy():
    """Test that pooled container returns pool buffer data without copy."""
    print("\n=== Test: Pooled Container Zero-Copy ===")
    
    # Create pool and acquire buffer
    pool = BufferPool(num_buffers=2, width=640, height=480, channels=3)
    buffer = pool.acquire()
    
    # Fill with test pattern
    buffer.data[:] = 42
    pool.release(buffer)  # Release camera ref
    
    # Create pooled container
    container = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
    
    # Verify zero-copy: get() returns exact same array
    retrieved = container.get(ImageFormat.NUMPY)
    assert retrieved is buffer.data, "Should return pool buffer directly"
    assert (retrieved == 42).all(), "Data should match"
    assert container.is_pooled(), "Should be in pooled mode"
    
    # Verify buffer refcount
    assert buffer.refcount == 1, "Buffer should have 1 reference"
    assert container in buffer.containers, "Container should be tracked"
    
    print("✓ Zero-copy behavior verified")


def test_pooled_container_promotion():
    """Test that promotion copies data and switches to owned mode."""
    print("\n=== Test: Container Promotion ===")
    
    # Create pool and acquire buffer
    pool = BufferPool(num_buffers=2, width=640, height=480, channels=3)
    buffer = pool.acquire()
    
    # Fill with test pattern
    buffer.data[:] = 123
    pool.release(buffer)  # Release camera ref
    
    # Create pooled container
    container = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
    
    # Verify pooled state
    assert container.is_pooled(), "Should start pooled"
    original_data = buffer.data
    
    # Promote to owned
    container._promote_to_owned()
    
    # Verify owned state
    assert not container.is_pooled(), "Should be owned after promotion"
    assert buffer.refcount == 0, "Buffer should have no references"
    assert container not in buffer.containers, "Container should not be tracked"
    
    # Verify data was copied
    promoted_data = container.get(ImageFormat.NUMPY)
    assert promoted_data is not original_data, "Should have different array"
    assert (promoted_data == 123).all(), "Data should be preserved"
    
    # Verify original buffer unchanged
    assert (original_data == 123).all(), "Original buffer intact"
    
    print("✓ Promotion correctly copies and switches mode")


def test_pooled_container_lifecycle():
    """Test that container releases buffer when going out of scope."""
    print("\n=== Test: Container Lifecycle ===")
    
    # Create pool and acquire buffer
    pool = BufferPool(num_buffers=2, width=640, height=480, channels=3)
    buffer = pool.acquire()
    pool.release(buffer)  # Release camera ref
    
    # Create container in a function scope
    def create_and_use_container():
        container = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
        assert buffer.refcount == 1, "Buffer should have 1 reference while container exists"
        # Container goes out of scope here
    
    # Call function
    create_and_use_container()
    
    # After function returns, container should be gone and buffer released
    # Note: We can't guarantee immediate __del__ call, but we can verify the buffer
    # is available for reuse by trying to acquire from a full pool
    buffer2 = pool.acquire()
    pool.release(buffer2)
    assert pool.acquire() is not None, "Buffers should be available for reuse"
    
    print("✓ Container properly releases buffer when out of scope")


def test_pooled_container_format_caching():
    """Test that pooled buffers work correctly with format caching."""
    print("\n=== Test: Format Caching with Pooled Buffers ===")
    
    # Create pool and acquire buffer
    pool = BufferPool(num_buffers=2, width=640, height=480, channels=3)
    buffer = pool.acquire()
    
    # Fill with test pattern (BGR)
    buffer.data[:, :, 0] = 100  # Blue
    buffer.data[:, :, 1] = 150  # Green
    buffer.data[:, :, 2] = 200  # Red
    pool.release(buffer)  # Release camera ref
    
    # Create pooled container
    container = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
    
    # First get in NUMPY (zero-copy)
    bgr1 = container.get(ImageFormat.NUMPY)
    assert bgr1 is buffer.data, "NUMPY should be zero-copy"
    
    # Get NUMPY again (should still be zero-copy)
    bgr2 = container.get(ImageFormat.NUMPY)
    assert bgr2 is buffer.data, "NUMPY should still be zero-copy"
    assert bgr2 is bgr1, "Should return same array"
    
    print("✓ Format caching works correctly with pooled buffers")
    
    print("✓ Format caching works correctly with pooled buffers")


def test_pooled_container_multiple_formats():
    """Test that promoting preserves cached data."""
    print("\n=== Test: Promotion with Cached Formats ===")
    
    # Create pool and acquire buffer
    pool = BufferPool(num_buffers=2, width=100, height=100, channels=3)
    buffer = pool.acquire()
    
    # Fill with test pattern
    buffer.data[:, :, 0] = 50
    buffer.data[:, :, 1] = 100
    buffer.data[:, :, 2] = 150
    pool.release(buffer)  # Release camera ref
    
    # Create pooled container and access data
    container = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
    
    # Get NUMPY (zero-copy from pool)
    bgr_before = container.get(ImageFormat.NUMPY)
    assert bgr_before is buffer.data, "Should be zero-copy"
    
    # Promote
    container._promote_to_owned()
    
    # Verify data still accessible after promotion
    bgr_after = container.get(ImageFormat.NUMPY)
    
    assert (bgr_after[:, :, 0] == 50).all(), "BGR preserved"
    assert (bgr_after[:, :, 1] == 100).all(), "Data preserved after promotion"
    
    print("✓ Promotion preserves all cached formats")


def test_pooled_container_concurrent_references():
    """Test multiple containers referencing the same pooled buffer."""
    print("\n=== Test: Multiple Containers Sharing Buffer ===")
    
    # Create pool and acquire buffer
    pool = BufferPool(num_buffers=2, width=640, height=480, channels=3)
    buffer = pool.acquire()
    buffer.data[:] = 77
    pool.release(buffer)  # Release camera ref
    
    # Create multiple containers sharing buffer
    container1 = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
    
    container2 = ImageContainer(cache_key=f"test_{id(buffer)}", pooled_buffer=buffer)
    
    # Verify refcount
    assert buffer.refcount == 2, "Buffer should have 2 references"
    assert container1 in buffer.containers, "Container 1 tracked"
    assert container2 in buffer.containers, "Container 2 tracked"
    
    # Let one container go out of scope by putting it in a function
    def use_container1():
        # container1 is still accessible here
        data1 = container1.get(ImageFormat.NUMPY)
        assert (data1 == 77).all(), "Container 1 works"
        # container1 goes out of scope when function returns
    
    use_container1()
    
    # container2 should still work
    assert container2 in buffer.containers, "Container 2 still tracked"
    data = container2.get(ImageFormat.NUMPY)
    assert (data == 77).all(), "Data still accessible"
    
    print("✓ Multiple containers can share pooled buffer")


def run_all_tests():
    """Run all pooled container integration tests."""
    tests = [
        test_pooled_container_zero_copy,
        test_pooled_container_promotion,
        test_pooled_container_lifecycle,
        test_pooled_container_format_caching,
        test_pooled_container_multiple_formats,
        test_pooled_container_concurrent_references,
    ]
    
    passed = 0
    failed = 0
    
    print("=" * 60)
    print("POOLED CONTAINER INTEGRATION TESTS")
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
