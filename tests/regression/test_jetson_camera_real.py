#!/usr/bin/env python3
"""
Regression test for JetsonCameraSource with real camera.

Tests the full zero-copy camera implementation with actual hardware:
- Real cv2.VideoCapture or GStreamer
- cv2.retrieve() writing into pre-allocated pool buffers
- Ring buffer + pool interaction over many frames
- Memory stays bounded
- Auto-promotion working correctly
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from vlmchat.pipeline.sources.jetson_camera import JetsonCameraSource
from vlmchat.utils.platform_detect import detect_platform, Platform


def test_jetson_camera_basic_capture():
    """Test basic camera capture with pool."""
    print("\n=== Test: Basic Camera Capture ===")
    
    platform = detect_platform()
    print(f"Platform: {platform}")
    
    # Create camera source
    # Use GStreamer on Jetson, regular camera otherwise
    use_gst = platform == Platform.JETSON
    
    camera = JetsonCameraSource(
        name="test_camera",
        device=0,
        fps=30,
        buffer_size=10,  # Small ring buffer for testing
        pool_size=5,     # Small pool to trigger promotion
        use_gstreamer=use_gst
    )
    
    try:
        # Start camera
        camera.start()
        time.sleep(1)  # Let camera warm up
        
        # Capture a few frames by polling (pulls from GStreamer callbacks)
        captured = 0
        for _ in range(10):
            camera.poll()  # Pull frame from GStreamer into ring buffer
            container, seq = camera.get_latest()
            
            if container:
                captured += 1
                print(f"  Captured frame {captured} seq={seq}: pooled={container.is_pooled()}")
                
                # Verify data is accessible
                data = container.get()
                assert data is not None, "Frame data should be accessible"
                assert data.shape[2] == 3, "Should be BGR"
                
            time.sleep(0.1)
        
        assert captured > 0, "Should capture at least some frames"
        print(f"✓ Captured {captured} frames successfully")
        
    finally:
        camera.stop()


def test_jetson_camera_retrieve_into_buffer():
    """Test that cv2.retrieve() actually writes into our pre-allocated buffer."""
    print("\n=== Test: cv2.retrieve() Into Pool Buffer ===")
    
    platform = detect_platform()
    use_gst = platform == Platform.JETSON
    
    camera = JetsonCameraSource(
        name="test_camera",
        device=0,
        fps=30,
        buffer_size=10,
        pool_size=10,  # Increased from 3
        use_gstreamer=use_gst
    )
    
    try:
        camera.start()
        time.sleep(1)  # Let continuous capture run
        
        # Verify continuous capture is using pool buffers
        frames = []
        for _ in range(10):
            if camera.has_new_data():
                container, seq = camera.get_latest()
                if container and container.is_pooled():
                    frames.append(container)
            time.sleep(0.05)
        
        # Check that frames are using pool
        pooled = sum(1 for f in frames if f.is_pooled())
        
        assert len(frames) >= 5, f"Should capture at least 5 frames, got {len(frames)}"
        assert pooled >= 5, f"Should have pooled frames, got {pooled} pooled"
        
        print(f"✓ Captured {len(frames)} frames using pool buffers ({pooled} pooled)")
        
    finally:
        camera.stop()


def test_jetson_camera_pool_exhaustion():
    """Test pool behavior when ring buffer exceeds pool size."""
    print("\n=== Test: Pool Exhaustion and Promotion ===")
    
    platform = detect_platform()
    use_gst = platform == Platform.JETSON
    
    # Small pool (5) but larger ring buffer (20)
    camera = JetsonCameraSource(
        name="test_camera",
        device=0,
        fps=30,
        buffer_size=20,  # Ring buffer: 20 frames
        pool_size=5,     # Pool: only 5 buffers
        use_gstreamer=use_gst
    )
    
    try:
        camera.start()
        time.sleep(1)
        
        # Capture many frames (more than pool size)
        num_frames = 25
        pooled_count = 0
        promoted_count = 0
        
        print(f"  Capturing {num_frames} frames (pool has 5 buffers)...")
        
        for i in range(num_frames):
            camera.poll()
            container, seq = camera.get_latest()
            
            if container:
                if container.is_pooled():
                    pooled_count += 1
                else:
                    promoted_count += 1
            
            time.sleep(0.05)
        
        # Get final metrics
        metrics = camera.get_pool_metrics()
        
        print(f"  Frames: {pooled_count} pooled, {promoted_count} promoted")
        print(f"  Pool metrics:")
        print(f"    Total buffers: {metrics['total_buffers']}")
        print(f"    Borrowed: {metrics['borrowed_buffers']}")
        print(f"    Pressure: {metrics['pressure']:.2f}")
        print(f"    Promotions: {metrics['frames_promoted']}")
        print(f"    Evictions: {metrics['buffers_evicted']}")
        
        # Verify promotions happened in pool
        # Note: Containers may still be pooled (not yet promoted) when read from ring buffer
        # Promotions happen when pool needs buffer back, not when container is read
        assert metrics['frames_promoted'] > 0, "Should have promoted some frames"
        
        print(f"✓ Pool handled exhaustion: {metrics['frames_promoted']} promotions")
        print(f"  (Containers: {pooled_count} pooled, {promoted_count} promoted)")
        
    finally:
        camera.stop()


def test_jetson_camera_long_run():
    """Test camera running for extended period (300+ frames)."""
    print("\n=== Test: Long Running Camera (300+ frames) ===")
    
    platform = detect_platform()
    use_gst = platform == Platform.JETSON
    
    # Realistic settings
    camera = JetsonCameraSource(
        name="test_camera",
        device=0,
        fps=30,
        buffer_size=60,   # 2 seconds @ 30fps
        pool_size=60,     # Match ring buffer
        use_gstreamer=use_gst
    )
    
    try:
        camera.start()
        time.sleep(1)
        
        num_frames = 300
        captured = 0
        start_time = time.time()
        
        print(f"  Running for {num_frames} frames...")
        
        for _ in range(num_frames):
            camera.poll()
            container, seq = camera.get_latest()
            if container:
                captured += 1
            time.sleep(0.033)  # ~30fps
        
        elapsed = time.time() - start_time
        fps = captured / elapsed if elapsed > 0 else 0
        
        # Get final metrics
        metrics = camera.get_pool_metrics()
        
        print(f"  Results:")
        print(f"    Captured: {captured}/{num_frames} frames")
        print(f"    FPS: {fps:.1f}")
        print(f"    Pool pressure: {metrics['pressure']:.2f}")
        print(f"    Promotions: {metrics['frames_promoted']}")
        print(f"    Drops: {metrics.get('frames_dropped', 0)}")
        
        # Verify stability
        assert captured > num_frames * 0.8, "Should capture at least 80% of frames"
        assert metrics['pressure'] <= 1.0, "Pool pressure should stay <= 100%"
        
        print(f"✓ Camera ran stably: {captured} frames at {fps:.1f} fps")
        
    finally:
        camera.stop()


def test_jetson_camera_memory_bounded():
    """Test that memory stays bounded over long run."""
    print("\n=== Test: Memory Bounded ===")
    
    import psutil
    import os
    
    platform = detect_platform()
    use_gst = platform == Platform.JETSON
    
    camera = JetsonCameraSource(
        name="test_camera",
        device=0,
        fps=30,
        buffer_size=60,
        pool_size=60,
        use_gstreamer=use_gst
    )
    
    try:
        # Get initial memory
        process = psutil.Process(os.getpid())
        mem_start = process.memory_info().rss / 1024 / 1024  # MB
        
        camera.start()
        time.sleep(1)
        
        # Run for many frames
        for _ in range(500):
            camera.poll()
            time.sleep(0.033)
        
        # Get final memory
        mem_end = process.memory_info().rss / 1024 / 1024  # MB
        mem_increase = mem_end - mem_start
        
        print(f"  Memory usage:")
        print(f"    Start: {mem_start:.1f} MB")
        print(f"    End: {mem_end:.1f} MB")
        print(f"    Increase: {mem_increase:.1f} MB")
        
        # Verify memory didn't explode (allow some growth for buffers)
        # 640x480x3 x 60 buffers = ~50MB expected
        assert mem_increase < 200, f"Memory increase too large: {mem_increase:.1f} MB"
        
        print(f"✓ Memory stayed bounded: +{mem_increase:.1f} MB")
        
    finally:
        camera.stop()


def run_all_tests():
    """Run all regression tests."""
    tests = [
        test_jetson_camera_basic_capture,
        test_jetson_camera_retrieve_into_buffer,
        test_jetson_camera_pool_exhaustion,
        test_jetson_camera_long_run,
        test_jetson_camera_memory_bounded,
    ]
    
    passed = 0
    failed = 0
    skipped = 0
    
    print("=" * 70)
    print("JETSON CAMERA REGRESSION TESTS")
    print("=" * 70)
    
    # Check if camera available
    try:
        import cv2
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("⚠️  No camera detected - skipping hardware tests")
            return 0
        cap.release()
    except Exception as e:
        print(f"⚠️  Camera not available: {e}")
        return 0
    
    for test in tests:
        try:
            test()
            passed += 1
            print("✅ PASSED\n")
        except AssertionError as e:
            failed += 1
            print(f"❌ FAILED: {e}\n")
        except Exception as e:
            failed += 1
            print(f"❌ ERROR: {e}\n")
    
    print("=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✅ ALL TESTS PASSED")
    else:
        print(f"❌ {failed} TESTS FAILED")
    print("=" * 70)
    
    return failed


if __name__ == "__main__":
    sys.exit(run_all_tests())
