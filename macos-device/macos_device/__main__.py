"""macOS device main entry point."""

import time
import logging
import sys
import threading
import concurrent.futures
from pathlib import Path
from camera_framework import (
    BaseTask,
    Runner, 
    Collector, 
    RateInstrument, 
    AvgDurationInstrument,
    MemoryInstrument,
    Buffer,
    drop_oldest_policy,
    blocking_policy,
    CocoCategory,
    PipelineTraverser,
    MermaidVisitor,
)
from camera_framework.config import CameraFrameworkConfig
from camera_framework.bridges import DeviceClient
from camera_framework.cameras import ImageLibraryCamera
from .camera import Camera
from .yolo_detector import YoloDetector
from .attribute_enricher import AttributeEnricher
from .category_router import YoloCategoryRouter
from .attribute_color_filter import AttributeColorFilter
from .clusterer import Clusterer
from .detection_tracker import DetectionTracker
from .alert_publisher import AlertPublisher
from .detection_filter import DetectionFilter
from .vlm_queue import VLMQueue
from .smolvlm_verifier import SmolVLMVerifier
from .config import MacOSDeviceConfig

# Configure root logger for console output
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s %(levelname)s [%(module)s] %(message)s',
    datefmt='%H:%M:%S'
)
# Our code at INFO; third-party libs stay at WARNING
logging.getLogger('macos_device').setLevel(logging.INFO)
logging.getLogger('camera_framework').setLevel(logging.INFO)

logger = logging.getLogger(__name__)


def main_standalone():
    """Run simple camera->display pipeline without MQTT."""
    logger.info("Starting camera pipeline (standalone mode)...")
    
    # Create metrics
    collector = Collector()
    collector.add_instrument(RateInstrument("pipeline.fps"), "pipeline.frame")
    collector.add_instrument(AvgDurationInstrument("pipeline.duration"), "pipeline.duration")
    
    # Add memory monitoring
    mem_instrument = MemoryInstrument("memory.objects", leak_threshold_seconds=30.0)
    collector.add_instrument(mem_instrument, "memory.track")
    collector._mem_instrument = mem_instrument
    
    # Create runner with collector
    runner = Runner(max_workers=4, collector=collector)
    
    # Create and add tasks - use ImageLibraryCamera
    camera = ImageLibraryCamera(
        image_dir="/Users/patrick/Dev/MOT15",  # MOT15 dataset
        width=1920,
        height=1080,
        framerate=1.0,  # 1 frame per second for debugging
        collector=collector,  # Pass collector for memory tracking
    )
    
    runner.add_task(camera)
    
    try:
        logger.info("Camera pipeline started (standalone mode)")
        logger.info("Press Ctrl+C to stop")
        
        while True:
            # Record frame
            collector.record("pipeline.frame", 1.0)
            
            # Time pipeline execution
            with collector.duration_timer("pipeline.duration"):
                runner.run_once()
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)  # 10ms - allows checking for ready tasks frequently
            
            # Print stats every 30 frames
            fps_stats = collector.get_stats("pipeline.fps")
            if fps_stats and fps_stats["count"] % 30 == 0:
                duration_stats = collector.get_stats("pipeline.duration")
                mem_stats = collector.get_stats("memory.objects")
                stats = runner.stats()
                
                log_msg = f"FPS: {fps_stats['rate']:.1f} | Avg Duration: {duration_stats['avg']:.1f}ms | Cache: {stats['cache']}"
                if mem_stats:
                    log_msg += f" | Mem: {mem_stats['total_alive']} alive, {mem_stats['total_cleaned']} cleaned"
                    if mem_stats['potential_leaks']:
                        log_msg += f" | ⚠️ LEAKS: {len(mem_stats['potential_leaks'])}"
                logger.info(log_msg)
    
    except KeyboardInterrupt:
        logger.info("Stopping...")
    
    finally:
        camera.stop()
        runner.shutdown()
        logger.info("Pipeline stopped")


def main_mqtt(diagram_only=False):
    """Run with MQTT integration (DeviceClient).
    
    Args:
        diagram_only: If True, build pipeline, generate diagram, and exit
    """
    
    logger.info("Starting camera pipeline (MQTT mode)..." if not diagram_only else "Building pipeline for diagram...")
    
    # Load configurations - will crash if files don't exist or are invalid
    framework_config_path = Path("camera_framework_config.yaml")
    device_config_path = Path("macos_device_config.yaml")
    
    logger.info(f"Loading camera framework config from: {framework_config_path}")
    framework_config = CameraFrameworkConfig.load(str(framework_config_path))
    
    logger.info(f"Loading macOS device config from: {device_config_path}")
    device_config = MacOSDeviceConfig.load(str(device_config_path))
    
    # Parse command line arguments
    schemas_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--schemas-path" and i + 1 < len(sys.argv):
            schemas_path = sys.argv[i + 1]
            break
    
    # Override MQTT schemas path if provided via CLI
    if schemas_path:
        device_config.sinks.mqtt.schemas_path = schemas_path
    
    # Create metrics
    collector = Collector()
    collector.add_instrument(RateInstrument("pipeline.fps"), "pipeline.frame")
    collector.add_instrument(AvgDurationInstrument("pipeline.duration"), "pipeline.duration")
    
    # Add memory monitoring
    mem_instrument = MemoryInstrument(
        "memory.objects", 
        leak_threshold_seconds=framework_config.pipeline.memory_leak_threshold
    )
    collector.add_instrument(mem_instrument, "memory.track")
    collector._mem_instrument = mem_instrument
    
    # Create runner with collector
    runner = Runner(max_workers=framework_config.pipeline.max_workers, collector=collector)
    
    # Create camera -> buffer -> YOLO pipeline
    camera = ImageLibraryCamera(
        image_dir=framework_config.sources.image_library.image_dir,
        width=framework_config.sources.image_library.width,
        height=framework_config.sources.image_library.height,
        framerate=framework_config.sources.image_library.framerate,
        collector=collector,
    )
    
    # Create buffer between camera and YOLO
    camera_buffer = Buffer(
        size=framework_config.buffers.default_size, 
        policy=drop_oldest_policy, 
        name="camera_to_yolo"
    )
    
    # Create YOLO task with config
    yolo = YoloDetector(config=device_config.tasks.yolo)
    
    # Create YOLO output buffer (router reads from this)
    yolo_buffer = Buffer(
        size=framework_config.buffers.default_size, 
        policy=drop_oldest_policy, 
        name="yolo_detections"
    )
    
    # Create shared detection filter
    filter_config = DetectionFilter()
    # Filters will be populated via MQTT commands from server
    
    # Create category router (filters to allowed categories only)
    router = YoloCategoryRouter(filter_config=filter_config)
    
    # Create buffer for filtered detections
    filtered_buffer = Buffer(size=5, policy=drop_oldest_policy, name="filtered_detections")
    
    # Create PersonAttribute task (enriches persons, passes through others)
    attributes = AttributeEnricher(
        config=device_config.tasks.attributes
    )
    
    # Create buffer for enriched detections
    enriched_buffer = Buffer(size=5, policy=drop_oldest_policy, name="enriched_detections")
    
    # Create attribute+color filter (extracts colors and filters by attributes+colors)
    attr_color_filter = AttributeColorFilter(
        config=device_config.tasks.color_filter,
        filter_config=filter_config
    )
    
    # Create buffer after color filtering
    color_filtered_buffer = Buffer(size=5, policy=drop_oldest_policy, name="color_filtered_detections", keep_latest=True)  # keep_latest for snapshot
    
    # Create clusterer (associates objects per filter)
    clusterer = Clusterer(
        config=device_config.tasks.clusterer,
        filter_config=filter_config
    )
    
    # Create buffer for clustered detections
    clustered_buffer = Buffer(size=5, policy=drop_oldest_policy, name="clustered_detections")
    
    # Create detection tracker (deduplicates and confirms alerts)
    tracker = DetectionTracker(
        config=device_config.tasks.tracker,
        filter_config=filter_config,
        vlm_max_attempts=device_config.tasks.smolvlm.vlm_max_attempts if device_config.tasks.smolvlm else 3,
    )
    
    # Create buffer for tracker output (goes to VLMQueue)
    tracker_buffer = Buffer(size=30, policy=drop_oldest_policy, name="tracker_output")
    
    # Create SmolVLM worker if configured
    smolvlm_worker = None
    if device_config.tasks.smolvlm:
        smolvlm_worker = SmolVLMVerifier(
            model_path=device_config.tasks.smolvlm.model_path,
            model_size=device_config.tasks.smolvlm.model_size,
            device_id=device_config.sinks.mqtt.device_id,
            max_new_tokens=device_config.tasks.smolvlm.max_new_tokens,
            result_callback=tracker.handle_vlm_result,
            name="smolvlm_worker",
        )
        logger.info(f"SmolVLM enabled: {device_config.tasks.smolvlm.model_path}")
    else:
        logger.info("SmolVLM disabled (not in config)")
    
    # Create VLMQueue (routes to immediate alerts or VLM verification)
    vlm_queue = VLMQueue(
        max_queue_size=10,
        smolvlm_worker=smolvlm_worker,
        name="vlm_queue",
    )
    
    # Create buffers for VLMQueue outputs
    alert_buffer = Buffer(size=30, policy=drop_oldest_policy, name="alert_buffer")
    vlm_buffer = Buffer(size=1, policy=blocking_policy, strict=True, name="vlm_buffer")
    
    # Wire pipeline: Camera -> YOLO -> CategoryRouter -> Attributes -> AttributeColorFilter -> Clusterer -> Tracker -> Alerts
    camera.add_output("frame", camera_buffer)
    yolo.add_input("frame", camera_buffer)
    yolo.add_output("detections", yolo_buffer)
    
    # Category router: filters by COCO category
    router.add_input("detections", yolo_buffer)
    router.add_output("filtered", filtered_buffer)
    
    # PersonAttributes: enriches persons, passes through others
    attributes.add_input("detections", filtered_buffer)
    attributes.add_output("enriched", enriched_buffer)
    
    # Attribute+color filter: extracts colors and filters by attributes+colors
    attr_color_filter.add_input("detections", enriched_buffer)
    attr_color_filter.add_output("filtered", color_filtered_buffer)
    
    # Clusterer: associates objects per filter
    clusterer.add_input("detections", color_filtered_buffer)
    clusterer.add_output("clustered", clustered_buffer)
    
    # Tracker: deduplicates and confirms
    tracker.add_input("clustered", clustered_buffer)
    tracker.add_output("confirmations", tracker_buffer)
    
    # VLMQueue: routes to immediate alerts or VLM verification
    vlm_queue.add_input("tracker", tracker_buffer)
    vlm_queue.add_output("alerts", alert_buffer)
    vlm_queue.add_output("vlm", vlm_buffer)
    
    # SmolVLM worker: processes VLM verifications and sends to alert_publisher (if enabled)
    if smolvlm_worker:
        smolvlm_worker.add_input("detections", vlm_buffer)
        smolvlm_worker.add_output("alerts", alert_buffer)
    
    # Add tasks to runner
    runner.add_task(camera)
    runner.add_task(yolo)
    runner.add_task(router)
    runner.add_task(attributes)
    runner.add_task(attr_color_filter)
    runner.add_task(clusterer)
    runner.add_task(tracker)
    runner.add_task(vlm_queue)
    if smolvlm_worker:
        runner.add_task(smolvlm_worker)

    # Create device client - adds MQTT tasks to runner
    client = DeviceClient(
        device_id=device_config.sinks.mqtt.device_id,
        device_type="macos",
        runner=runner,
        broker_host=device_config.sinks.mqtt.broker_host,
        broker_port=device_config.sinks.mqtt.broker_port,
        schemas_path=device_config.sinks.mqtt.schemas_path,
        detection_filter=filter_config,  # Pass filter for MQTT updates
        sample_buffer=color_filtered_buffer,  # Snapshot samples after attribute filter
    )
    
    # Create alert publisher (publishes confirmed detections)
    alert_publisher = AlertPublisher(
        device_id=device_config.sinks.mqtt.device_id,
        mqtt_client=client.mqtt_client,
    )
    alert_publisher.add_input("alerts", alert_buffer)
    runner.add_task(alert_publisher)
    
    # If diagram_only mode, generate diagram and exit
    if diagram_only:
        logger.debug(f"Runner has {len(runner.tasks)} tasks")
        for i, task in enumerate(runner.tasks):
            logger.debug(f"  Task {i}: {task.name} - inputs={len(task.inputs)}, outputs={len(task.outputs)}")
        
        visitor = MermaidVisitor()
        traverser = PipelineTraverser(runner)
        traverser.traverse(visitor)
        mermaid_code = visitor.get_result()
        
        # Write Mermaid source to local directory
        output_dir = Path("diagrams")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        mmd_file = output_dir / "pipeline-topology.mmd"
        with open(mmd_file, 'w') as f:
            f.write(mermaid_code)
        
        logger.info(f"✅ Mermaid diagram written to {mmd_file}")
        logger.info(f"Run 'just diagram' to compile and publish to web UI")
        return

    # ── Lifecycle: warmup (blocking model loads, run in parallel) ────────────
    # t.__class__.warmup is not BaseTask.warmup detects tasks that actually
    # override warmup() — no-op base calls are skipped from the log/parallel pool
    # but ALL tasks get warmup() called (safe: base is a no-op).
    warmup_tasks = [t for t in runner.tasks if t.__class__.warmup is not BaseTask.warmup]
    if warmup_tasks:
        logger.info(f"Warming up {len(warmup_tasks)} task(s) in parallel: {[t.name for t in warmup_tasks]}")
        errors: list[tuple[str, Exception]] = []
        lock = threading.Lock()

        def _warmup(task: BaseTask) -> None:
            t0 = time.time()
            logger.info(f"  [{task.name}] warmup starting...")
            try:
                task.warmup()
                logger.info(f"  [{task.name}] warmup done in {time.time() - t0:.1f}s")
            except Exception as e:
                logger.error(f"  [{task.name}] warmup FAILED: {e}", exc_info=True)
                with lock:
                    errors.append((task.name, e))

        with concurrent.futures.ThreadPoolExecutor(max_workers=len(warmup_tasks)) as pool:
            futs = {pool.submit(_warmup, t): t for t in warmup_tasks}
            concurrent.futures.wait(futs)  # BLOCKS until all warmups finish

        if errors:
            raise RuntimeError(f"Warmup failed for: {[name for name, _ in errors]}")

        logger.info("All warmups complete — pipeline ready")
    else:
        logger.info("No warmup tasks — pipeline ready")

    # ── Lifecycle: start (background threads / connections) ───────────────────
    for task in runner.tasks:
        task.start()

    try:
        logger.info("Connecting to MQTT broker...")
        client.start()  # Connect MQTT, register device
        
        logger.info("Pipeline running (press Ctrl+C to stop)...")
        # Main loop - runner handles all tasks
        while True:
            collector.record("pipeline.frame", 1.0)
            
            with collector.duration_timer("pipeline.duration"):
                runner.run_once()
            
            # Small sleep to prevent CPU spinning
            time.sleep(0.01)  # 10ms - allows checking for ready tasks frequently
            
            # Print stats occasionally
            fps_stats = collector.get_stats("pipeline.fps")
            if fps_stats and fps_stats["count"] % 30 == 0:
                duration_stats = collector.get_stats("pipeline.duration")
                mem_stats = collector.get_stats("memory.objects")
                stats = runner.stats()
                
                log_msg = f"FPS: {fps_stats['rate']:.1f} | Avg Duration: {duration_stats['avg']:.1f}ms"
                if 'cache' in stats:
                    log_msg += f""
                if 'cache' in stats:
                    log_msg += f" | Cache: {stats['cache']}"
                if mem_stats:
                    log_msg += f" | Mem: {mem_stats['total_alive']} alive, {mem_stats['total_cleaned']} cleaned"
                    if mem_stats['potential_leaks']:
                        # Count total leaked objects across all types
                        total_leaked = sum(leak['count'] for leak in mem_stats['potential_leaks'])
                        log_msg += f" | ⚠️ LEAKS: {total_leaked} objects ({len(mem_stats['potential_leaks'])} types)"
                        # Log leak details
                        for leak in mem_stats['potential_leaks']:
                            logger.warning(f"Potential leak: {leak['count']} {leak['type']} objects, "
                                         f"max age {leak['max_age_sec']:.1f}s, "
                                         f"total size {leak['total_size_bytes']} bytes")
                logger.info(log_msg)
        
    except KeyboardInterrupt:
        logger.info("Stopping...")
    finally:
        runner.shutdown(wait=True)
        for task in reversed(runner.tasks):
            task.stop()
        client.stop()
        logger.info("Pipeline stopped")


def main_diagram():
    """Generate pipeline diagram as Mermaid source."""
    # Just call main_mqtt with diagram_only=True to build the real pipeline
    main_mqtt(diagram_only=True)



def main():
    """Entry point - choose mode based on command line args."""
    import sys
    
    if "--diagram" in sys.argv:
        main_diagram()
    elif "--mqtt" in sys.argv:
        print("Usage: python -m macos_device --mqtt [--schemas-path PATH]")
        print("  --mqtt: Enable MQTT integration")
        print("  --schemas-path: Path to schemas directory (default: ../../shared/schemas)")
        print("")
        main_mqtt()
    else:
        print("Usage: python -m macos_device [OPTIONS]")
        print("")
        print("Options:")
        print("  --mqtt              Enable MQTT integration (full pipeline)")
        print("  --diagram           Generate pipeline diagram (Mermaid + SVG)")
        print("  (no flags)          Run standalone mode (camera only)")
        print("")
        print("Examples:")
        print("  python -m macos_device                # Standalone camera")
        print("  python -m macos_device --mqtt         # Full MQTT pipeline")
        print("  python -m macos_device --diagram      # Generate diagram")
        print("")
        main_standalone()


if __name__ == "__main__":
    main()
