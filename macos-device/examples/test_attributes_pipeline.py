"""Example: Camera -> YOLO -> PersonAttributes pipeline."""

import logging
from camera_framework import Runner, Collector, RateInstrument, AvgDurationInstrument, Buffer, drop_oldest_policy
from camera_framework.cameras import ImageLibraryCamera
from macos_device import YoloTask, PersonAttributeTask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Run camera -> YOLO -> attributes pipeline."""
    
    logger.info("Starting camera -> YOLO -> attributes pipeline...")
    
    # Create metrics collector
    collector = Collector()
    collector.add_instrument(RateInstrument("pipeline.fps"), "pipeline.frame")
    collector.add_instrument(AvgDurationInstrument("pipeline.duration"), "pipeline.duration")
    
    # Create runner
    runner = Runner(max_workers=4, collector=collector)
    
    # Create camera (ImageLibrary for testing)
    camera = ImageLibraryCamera(
        image_dir="/Users/patrick/Dev/MOT15",
        width=1920,
        height=1080,
        framerate=5.0,
        collector=collector,
    )
    
    # Create buffers
    camera_buffer = Buffer(size=5, policy=drop_oldest_policy, name="camera_to_yolo")
    yolo_buffer = Buffer(size=5, policy=drop_oldest_policy, name="yolo_to_attributes")
    attributes_buffer = Buffer(size=1, policy=drop_oldest_policy, name="attributes_output")
    
    # Create YOLO task
    yolo = YoloTask(
        model_path="yolov8n.pt",
        confidence=0.25,
    )
    
    # Create PersonAttribute task
    attributes = PersonAttributeTask(
        model_path="/Users/patrick/Downloads/pa_model_best_v3.onnx"
    )
    
    # Wire pipeline: Camera -> YOLO -> Attributes
    camera.add_output("frame", camera_buffer)
    yolo.add_input("frame", camera_buffer)
    yolo.add_output("detections", yolo_buffer)
    attributes.add_input("detections", yolo_buffer)
    attributes.add_output("enriched", attributes_buffer)
    
    # Add tasks to runner
    runner.add_task(camera)
    runner.add_task(yolo)
    runner.add_task(attributes)
    
    # Start camera background thread
    camera.start()
    
    try:
        logger.info("Pipeline running (press Ctrl+C to stop)...")
        
        frame_count = 0
        while True:
            collector.record("pipeline.frame", 1.0)
            
            with collector.duration_timer("pipeline.duration"):
                runner.run_once()
            
            frame_count += 1
            
            # Print stats every 30 frames
            if frame_count % 30 == 0:
                fps_stats = collector.get_stats("pipeline.fps")
                duration_stats = collector.get_stats("pipeline.duration")
                stats = runner.stats()
                
                logger.info(
                    f"FPS: {fps_stats['rate']:.1f} | "
                    f"Avg Duration: {duration_stats['avg']:.1f}ms | "
                    f"Cache: {stats['cache']}"
                )
                
                # Show detection results with attributes
                if attributes_buffer.has_data():
                    detections = attributes_buffer.get()
                    if detections:
                        logger.info(f"\n=== Frame {frame_count} Detections ===")
                        for i, det in enumerate(detections):
                            metadata = det.metadata if hasattr(det, 'metadata') and det.metadata else {}
                            
                            # Basic detection info
                            category = metadata.get('category', 'unknown')
                            bbox = det.bbox
                            conf = det.confidence
                            
                            logger.info(f"Detection {i+1}: {category} @ ({bbox[0]:.0f},{bbox[1]:.0f}) conf={conf:.2f}")
                            
                            # Show attributes if present
                            if 'attributes' in metadata:
                                attrs = metadata['attributes']
                                
                                # Show active attributes (value=True)
                                active_attrs = [k for k, v in attrs.items() if v['value']]
                                if active_attrs:
                                    logger.info(f"  Attributes: {', '.join(active_attrs)}")
                                
                                # Show top 5 by confidence
                                top_attrs = sorted(attrs.items(), key=lambda x: x[1]['confidence'], reverse=True)[:5]
                                logger.info(f"  Top confidence: {', '.join(f'{k}={v[\"confidence\"]:.2f}' for k, v in top_attrs)}")
                        
                        logger.info("=" * 40)
    
    except KeyboardInterrupt:
        logger.info("Stopping...")
    
    finally:
        runner.shutdown(wait=True)
        camera.stop()
        logger.info("Pipeline stopped")


if __name__ == "__main__":
    main()
