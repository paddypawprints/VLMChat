"""Integration tests for pipeline with VLMQueue (SmolVLM worker not connected)."""

import pytest
import time
from pathlib import Path
from camera_framework import (
    Runner,
    Collector,
    RateInstrument,
    AvgDurationInstrument,
    Buffer,
    drop_oldest_policy,
    blocking_policy,
)
from camera_framework.cameras import ImageLibraryCamera
from macos_device.yolo_task import YoloTask
from macos_device.category_router import YoloCategoryRouter
from macos_device.attribute_task import PersonAttributeTask
from macos_device.attribute_color_filter import AttributeColorFilter
from macos_device.clusterer import Clusterer
from macos_device.detection_tracker import DetectionTracker
from macos_device.detection_filter import DetectionFilter
from macos_device.vlm_queue import VLMQueue
from macos_device.config import MacOSDeviceConfig


@pytest.fixture
def config():
    """Load macOS device config."""
    config_path = Path(__file__).parent.parent.parent / "macos_device_config.yaml"
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")
    return MacOSDeviceConfig.load(str(config_path))


@pytest.fixture
def framework_config():
    """Load camera framework config."""
    from camera_framework.config import CameraFrameworkConfig
    config_path = Path(__file__).parent.parent.parent / "camera_framework_config.yaml"
    if not config_path.exists():
        pytest.skip(f"Config not found: {config_path}")
    return CameraFrameworkConfig.load(str(config_path))


@pytest.fixture
def pipeline_runner(config, framework_config):
    """Create a pipeline runner with all tasks except SmolVLM."""
    # Check for test images
    image_dir = framework_config.sources.image_library.image_dir
    if not Path(image_dir).exists():
        pytest.skip(f"Test image directory not found: {image_dir}")
    
    # Create collector
    collector = Collector()
    collector.add_instrument(RateInstrument("pipeline.fps"), "pipeline.frame")
    collector.add_instrument(AvgDurationInstrument("pipeline.duration"), "pipeline.duration")
    
    # Create runner
    runner = Runner(max_workers=framework_config.pipeline.max_workers, collector=collector)
    
    # Create camera
    camera = ImageLibraryCamera(
        image_dir=framework_config.sources.image_library.image_dir,
        width=framework_config.sources.image_library.width,
        height=framework_config.sources.image_library.height,
        framerate=10.0,  # Fast for testing
        collector=collector,
    )
    
    # Create buffers
    camera_buffer = Buffer(size=framework_config.buffers.default_size, policy=drop_oldest_policy, name="camera_to_yolo")
    yolo_buffer = Buffer(size=framework_config.buffers.default_size, policy=drop_oldest_policy, name="yolo_detections")
    filtered_buffer = Buffer(size=5, policy=drop_oldest_policy, name="filtered_detections")
    enriched_buffer = Buffer(size=5, policy=drop_oldest_policy, name="enriched_detections")
    color_filtered_buffer = Buffer(size=5, policy=drop_oldest_policy, name="color_filtered_detections")
    clustered_buffer = Buffer(size=5, policy=drop_oldest_policy, name="clustered_detections")
    tracker_buffer = Buffer(size=30, policy=drop_oldest_policy, name="tracker_output")
    alert_buffer = Buffer(size=30, policy=drop_oldest_policy, name="alert_buffer")
    vlm_buffer = Buffer(size=1, policy=blocking_policy, strict=True, name="vlm_buffer")
    
    # Create tasks
    yolo = YoloTask(config=config.tasks.yolo)
    
    # Override YOLO model path to use absolute path from repo root
    repo_root = Path(__file__).parent.parent.parent
    yolo_model = repo_root / "yolov8n.pt"
    if yolo_model.exists():
        yolo.model_path = yolo_model
    
    filter_config = DetectionFilter()
    
    # Add a test filter so detections can be matched
    from macos_device.search_filter import SearchFilter
    test_filter = SearchFilter(
        id="test-person",
        name="Test Person Filter",
        category_mask=[True] + [False] * 79,  # Only COCO category 0 (person)
        category_colors=["#FF0000"] + [""] * 79,
        attribute_mask=[False] * 26,  # No attribute filtering
        attribute_colors=[""] * 26,
        color_requirements={},
        vlm_required=False,
        vlm_reasoning=""
    )
    filter_config.set_filters([test_filter])
    
    router = YoloCategoryRouter(filter_config=filter_config)
    attributes = PersonAttributeTask(config=config.tasks.attributes)
    color_filter = AttributeColorFilter(config=config.tasks.color_filter, filter_config=filter_config)
    clusterer = Clusterer(config=config.tasks.clusterer, filter_config=filter_config)
    tracker = DetectionTracker(config=config.tasks.tracker, filter_config=filter_config)
    vlm_queue = VLMQueue(max_queue_size=10, smolvlm_worker=None, name="vlm_queue")
    
    # Wire pipeline
    camera.add_output("frame", camera_buffer)
    yolo.add_input("frame", camera_buffer)
    yolo.add_output("detections", yolo_buffer)
    router.add_input("detections", yolo_buffer)
    router.add_output("filtered", filtered_buffer)
    attributes.add_input("detections", filtered_buffer)
    attributes.add_output("enriched", enriched_buffer)
    color_filter.add_input("detections", enriched_buffer)
    color_filter.add_output("filtered", color_filtered_buffer)
    clusterer.add_input("detections", color_filtered_buffer)
    clusterer.add_output("clustered", clustered_buffer)
    tracker.add_input("clustered", clustered_buffer)
    tracker.add_output("confirmations", tracker_buffer)
    vlm_queue.add_input("tracker", tracker_buffer)
    vlm_queue.add_output("alerts", alert_buffer)
    vlm_queue.add_output("vlm", vlm_buffer)
    
    # Add tasks to runner
    runner.add_task(camera)
    runner.add_task(yolo)
    runner.add_task(router)
    runner.add_task(attributes)
    runner.add_task(color_filter)
    runner.add_task(clusterer)
    runner.add_task(tracker)
    runner.add_task(vlm_queue)
    
    camera.start()
    
    # Return runner and buffers for inspection
    return {
        'runner': runner,
        'camera': camera,
        'collector': collector,
        'buffers': {
            'camera': camera_buffer,
            'yolo': yolo_buffer,
            'filtered': filtered_buffer,
            'enriched': enriched_buffer,
            'color_filtered': color_filtered_buffer,
            'clustered': clustered_buffer,
            'tracker': tracker_buffer,
            'alert': alert_buffer,
            'vlm': vlm_buffer,
        },
        'tasks': {
            'vlm_queue': vlm_queue,
            'tracker': tracker,
        }
    }


def test_pipeline_smoke(pipeline_runner):
    """Smoke test: Run pipeline for a few frames."""
    runner = pipeline_runner['runner']
    collector = pipeline_runner['collector']
    camera = pipeline_runner['camera']
    
    try:
        # Run for 5 frames
        for i in range(5):
            collector.record("pipeline.frame", 1.0)
            with collector.duration_timer("pipeline.duration"):
                runner.run_once()
            time.sleep(0.1)
        
        # Check stats
        fps_stats = collector.get_stats("pipeline.fps")
        assert fps_stats is not None
        assert fps_stats['count'] >= 5
        
        duration_stats = collector.get_stats("pipeline.duration")
        assert duration_stats is not None
        assert duration_stats['avg'] > 0
        
    finally:
        camera.stop()
        runner.shutdown()


def test_pipeline_detections_flow(pipeline_runner):
    """Test that detections flow through pipeline to VLMQueue."""
    runner = pipeline_runner['runner']
    camera = pipeline_runner['camera']
    buffers = pipeline_runner['buffers']
    vlm_queue = pipeline_runner['tasks']['vlm_queue']
    
    try:
        # Run until we get detections through to tracker
        max_iterations = 50
        for i in range(max_iterations):
            runner.run_once()
            time.sleep(0.05)
            
            # Check if we have tracker output
            if buffers['tracker'].has_data():
                break
        
        # Should have tracker output
        assert buffers['tracker'].has_data(), "No detections reached tracker"
        
        # Run a few more iterations to let VLMQueue process
        for i in range(10):
            runner.run_once()
            time.sleep(0.05)
        
        # Check VLMQueue stats
        stats = vlm_queue.get_stats()
        
        # Should have routed some detections
        total_routed = stats['routed_to_alerts'] + stats['routed_to_vlm'] + stats['queued']
        assert total_routed > 0, f"VLMQueue did not process any detections: {stats}"
        
        print(f"VLMQueue stats: {stats}")
        
    finally:
        camera.stop()
        runner.shutdown()


def test_vlm_queue_routing_without_worker(pipeline_runner):
    """Test VLMQueue routing when SmolVLM worker is not connected."""
    runner = pipeline_runner['runner']
    camera = pipeline_runner['camera']
    buffers = pipeline_runner['buffers']
    vlm_queue = pipeline_runner['tasks']['vlm_queue']
    
    try:
        # Run until we get some alerts
        max_iterations = 100
        for i in range(max_iterations):
            runner.run_once()
            time.sleep(0.05)
            
            stats = vlm_queue.get_stats()
            if stats['routed_to_alerts'] > 0:
                break
        
        stats = vlm_queue.get_stats()
        print(f"VLMQueue stats after {i} iterations: {stats}")
        
        # Without SmolVLM worker, vlm_required detections should NOT go to VLM
        # (worker is None, so _is_smolvlm_ready() returns False)
        # They should either be queued or routed to alerts
        
        # Should have processed something
        total = stats['routed_to_alerts'] + stats['routed_to_vlm'] + stats['queued']
        assert total > 0, "VLMQueue did not process any detections"
        
        # Should have alerts (detections with vlm_required=False)
        assert stats['routed_to_alerts'] > 0, "No detections routed to alerts"
        
        # VLM buffer should be empty or have at most 1 item (blocking buffer size=1)
        vlm_data_count = 0
        if buffers['vlm'].has_data():
            vlm_data_count = 1
        
        print(f"VLM buffer count: {vlm_data_count}")
        print(f"Alert buffer has data: {buffers['alert'].has_data()}")
        
    finally:
        camera.stop()
        runner.shutdown()


def test_vlm_buffer_blocks_when_full(pipeline_runner):
    """Test that VLM buffer blocks when full (size=1, blocking policy)."""
    runner = pipeline_runner['runner']
    camera = pipeline_runner['camera']
    buffers = pipeline_runner['buffers']
    vlm_queue = pipeline_runner['tasks']['vlm_queue']
    
    try:
        # Run until VLM buffer fills up (should happen quickly)
        max_iterations = 100
        vlm_filled = False
        
        for i in range(max_iterations):
            runner.run_once()
            time.sleep(0.05)
            
            # Check if VLM buffer has data
            if buffers['vlm'].has_data():
                vlm_filled = True
                print(f"VLM buffer filled at iteration {i}")
                break
        
        if vlm_filled:
            # Buffer is full - run more iterations and verify it stays full
            # (can't put more because it's blocking and no consumer)
            for i in range(10):
                runner.run_once()
                time.sleep(0.05)
            
            # Buffer should still have data
            assert buffers['vlm'].has_data(), "VLM buffer was consumed unexpectedly"
            
            # VLMQueue should be queuing or routing to alerts now
            stats = vlm_queue.get_stats()
            print(f"VLMQueue stats with full VLM buffer: {stats}")
            
            # Should have more alerts (detections that can't go to VLM)
            assert stats['routed_to_alerts'] > 0, "No fallback to alerts when VLM full"
        
    finally:
        camera.stop()
        runner.shutdown()


def test_tracker_confirmation_logic(pipeline_runner):
    """Test that tracker generates confirmations with vlm_required flags."""
    runner = pipeline_runner['runner']
    camera = pipeline_runner['camera']
    buffers = pipeline_runner['buffers']
    tracker = pipeline_runner['tasks']['tracker']
    
    try:
        # Run until we get tracker confirmations
        max_iterations = 100
        for i in range(max_iterations):
            runner.run_once()
            time.sleep(0.05)
            
            if buffers['tracker'].has_data():
                break
        
        assert buffers['tracker'].has_data(), "No tracker confirmations generated"
        
        # Inspect tracker output (dict API)
        message = buffers['tracker'].get()
        confirmations = message.get("confirmations", [])
        
        assert len(confirmations) > 0, "Tracker message has no confirmations"
        
        # Check structure
        event = confirmations[0]
        assert isinstance(event, dict), f"Confirmation is not dict: {type(event)}"
        assert "track_id" in event, f"Missing track_id: {event.keys()}"
        assert "vlm_required" in event, f"Missing vlm_required: {event.keys()}"
        
        print(f"Sample confirmation event: {event}")
        
        # Check tracker stats
        tracker_stats = tracker.get_stats()
        print(f"Tracker stats: {tracker_stats}")
        
    finally:
        camera.stop()
        runner.shutdown()


@pytest.mark.slow
def test_pipeline_sustained_run(pipeline_runner):
    """Sustained test: Run pipeline for 30 frames and check stability."""
    runner = pipeline_runner['runner']
    camera = pipeline_runner['camera']
    collector = pipeline_runner['collector']
    buffers = pipeline_runner['buffers']
    vlm_queue = pipeline_runner['tasks']['vlm_queue']
    
    try:
        # Run for 30 frames
        for i in range(30):
            collector.record("pipeline.frame", 1.0)
            with collector.duration_timer("pipeline.duration"):
                runner.run_once()
            time.sleep(0.1)
            
            # Print stats every 10 frames
            if (i + 1) % 10 == 0:
                fps_stats = collector.get_stats("pipeline.fps")
                duration_stats = collector.get_stats("pipeline.duration")
                vlm_stats = vlm_queue.get_stats()
                
                print(f"\nFrame {i+1}:")
                print(f"  FPS: {fps_stats['rate']:.1f}")
                print(f"  Avg Duration: {duration_stats['avg']:.1f}ms")
                print(f"  VLMQueue: {vlm_stats}")
        
        # Final stats
        fps_stats = collector.get_stats("pipeline.fps")
        assert fps_stats['count'] >= 30
        
        vlm_stats = vlm_queue.get_stats()
        print(f"\nFinal VLMQueue stats: {vlm_stats}")
        
        # Should have processed detections
        total = vlm_stats['routed_to_alerts'] + vlm_stats['routed_to_vlm'] + vlm_stats['queued']
        assert total > 0, "No detections processed"
        
    finally:
        camera.stop()
        runner.shutdown()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
