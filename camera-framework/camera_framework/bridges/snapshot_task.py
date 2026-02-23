"""Snapshot capture task - standalone sink that samples from buffer on MQTT command."""

import logging
import base64
import io
from datetime import datetime
from pathlib import Path
from ..task import BaseTask
from ..buffer import Buffer

logger = logging.getLogger(__name__)


class SnapshotTask(BaseTask):
    """Standalone sink task that captures and publishes snapshots on MQTT command.
    
    Samples (peeks) from a configured buffer when snapshot command arrives.
    Does not connect to pipeline inputs/outputs - operates independently.
    """
    
    def __init__(self, mqtt_client, device_id: str, sample_buffer: Buffer, defer_subscription: bool = False):
        """
        Initialize snapshot task.
        
        Args:
            mqtt_client: MQTT client for publishing snapshots (can be None if deferred)
            device_id: Device identifier
            sample_buffer: Buffer to sample from (peek without consuming)
            defer_subscription: If True, don't subscribe in __init__ (caller will subscribe manually)
        """
        super().__init__(name=f"SnapshotTask-{device_id}")
        self.mqtt_client = mqtt_client
        self.device_id = device_id
        self.sample_buffer = sample_buffer
        self.snapshot_requested = False
        
        # Register as observer on the buffer
        sample_buffer.add_observer(self, "snapshot")
        
        # Subscribe to snapshot commands (unless deferred or mqtt_client is None)
        if not defer_subscription and mqtt_client is not None:
            command_topic = f"devices/{device_id}/commands/snapshot"
            self.mqtt_client.subscribe(command_topic, self._handle_snapshot_command)
            logger.info(f"SnapshotTask initialized, subscribed to {command_topic}")
    
    def _handle_snapshot_command(self, topic: str, payload: dict):
        """Handle incoming snapshot request."""
        logger.info(f"🎯 SNAPSHOT COMMAND RECEIVED on {topic}: {payload}")
        logger.info(f"🎯 Setting snapshot_requested flag to True")
        # Set flag to capture on next process() call
        self.snapshot_requested = True
        logger.info(f"🎯 Flag set successfully: snapshot_requested={self.snapshot_requested}")
    
    def is_ready(self) -> bool:
        """Task is ready when snapshot requested or on periodic basis."""
        # Always ready - we check snapshot_requested inside process()
        # This ensures process() is called regularly to check the flag
        return True
    
    def process(self) -> None:
        """
        Check if snapshot requested and capture from sample buffer.
        
        This is a sink task - it doesn't process pipeline data.
        Only samples from buffer when MQTT command is received.
        """
        try:
            # Debug: Log every 100th call to confirm process() is being called
            if not hasattr(self, '_process_count'):
                self._process_count = 0
            self._process_count += 1
            
            if self._process_count % 100 == 0:
                logger.debug(f"SnapshotTask.process() called {self._process_count} times, snapshot_requested={self.snapshot_requested}")
            
            # If snapshot requested, capture and publish
            if self.snapshot_requested:
                logger.info(f"📸 Snapshot flag detected! Capturing...")
                self._capture_and_publish_snapshot()
                self.snapshot_requested = False
                logger.info(f"📸 Snapshot flag cleared")
            
        except Exception as e:
            logger.error(f"Error in SnapshotTask.process: {e}")
            raise  # Propagate exception - snapshot capture is critical
    
    def _capture_and_publish_snapshot(self):
        """Capture from sample buffer and publish as snapshot."""
        try:
            # Peek (copy without consuming) from sample buffer - returns last processed value
            ctx_data = self.sample_buffer.peek()
            
            if not ctx_data:
                logger.warning("❌ No data available in sample buffer for snapshot (peek returned None)")
                return
            
            # Extract detections from context
            detections = ctx_data.get("detections", []) if isinstance(ctx_data, dict) else []
            
            logger.info(f"📋 ctx_data type: {type(ctx_data)}, is dict: {isinstance(ctx_data, dict)}")
            if isinstance(ctx_data, dict):
                logger.info(f"📋 ctx_data keys: {list(ctx_data.keys())}")
                logger.info(f"📋 detections type: {type(detections)}, count: {len(detections) if detections else 0}")
            
            if not detections:
                logger.warning("❌ No detections available in sample buffer")
                logger.warning(f"❌ ctx_data: {ctx_data}")
                return
            
            # Get frame from first detection
            frame = None
            if detections and detections[0].source_image:
                frame = detections[0].source_image
            
            if frame is None:
                logger.warning("No frame available for snapshot")
                return
            
            logger.info(f"Capturing snapshot: {frame.size[0]}x{frame.size[1]} pixels with {len(detections)} detections")
            
            # Save YOLO crops to disk for debugging
            self._save_detection_crops(frame, detections)
            
            # Convert to JPEG and encode as base64
            buffer = io.BytesIO()
            frame.save(buffer, format='JPEG', quality=85)
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
            
            logger.info(f"Image encoded: {len(image_base64)} base64 chars ({len(image_base64)//1024}KB)")
            
            # Get image dimensions
            width, height = frame.size
            
            # Add detection bounding boxes if available
            # Note: Synthetic full-frame detections (conf=0.0) are kept in the detections list
            # to preserve the image, but we don't include them in detection_boxes since
            # they would draw a box around the entire frame
            detection_boxes = []
            if detections:
                for det in detections:
                    logger.info(f"Detection attributes: bbox={det.bbox}, conf={det.confidence}, cat={det.category}, has_source={det.source_image is not None}")
                    
                    # Skip synthetic full-frame detections in bounding box overlay
                    if det.confidence < 0.01:
                        logger.info(f"Skipping synthetic full-frame detection in overlay (conf={det.confidence})")
                        continue
                    
                    detection_boxes.append({
                        "bbox": list(det.bbox),
                        "confidence": det.confidence,
                        "category": det.category.label,
                        "category_id": det.category.id
                    })
                logger.info(f"Added {len(detection_boxes)} detections to snapshot overlay")
            else:
                logger.warning("No detections available for snapshot")
            
            # Prepare snapshot message
            snapshot_topic = f"devices/{self.device_id}/snapshot"
            snapshot_message = {
                "device_id": self.device_id,
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "image": image_base64,
                "format": "jpeg",
                "width": width,
                "height": height,
                "detections": detection_boxes
            }
            
            logger.info(f"Publishing snapshot to topic: {snapshot_topic} with {len(detection_boxes)} detections")
            # Publish snapshot
            self.mqtt_client.publish(snapshot_topic, snapshot_message, qos=1)
            
            logger.info(f"Snapshot published successfully: {width}x{height}, {len(image_base64)} bytes")
            
        except Exception as e:
            logger.error(f"Snapshot capture failed: {type(e).__name__}: {e}", exc_info=True)
    
    def _save_detection_crops(self, frame, detections):
        """Save YOLO detection crops to disk for debugging color extraction.
        
        Args:
            frame: PIL Image (source frame)
            detections: List of Detection objects
        """
        try:
            # Create crops directory in /tmp
            crops_dir = Path("/tmp/vlmchat_crops")
            crops_dir.mkdir(parents=True, exist_ok=True)
            
            # Timestamp for this snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            logger.info(f"💾 Saving {len(detections)} detection crops to {crops_dir}/")
            
            for i, det in enumerate(detections):
                # Skip synthetic full-frame detections (confidence=0.0)
                if det.confidence < 0.01:
                    continue
                
                # Crop the detection region
                x1, y1, x2, y2 = det.bbox
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2, y2 = min(frame.width, int(x2)), min(frame.height, int(y2))
                
                if x2 <= x1 or y2 <= y1:
                    logger.warning(f"Invalid bbox for detection {i}: {det.bbox}")
                    continue
                
                crop = frame.crop((x1, y1, x2, y2))
                
                # Get attributes if available
                attrs = det.metadata.get('attributes', {})
                attr_str = "_".join([name for name, data in attrs.items() if data.get('value', False)])
                if not attr_str:
                    attr_str = "noattrs"
                
                # Get extracted colors if available
                colors = det.metadata.get('colors', {})
                color_str = "_".join([f"{k}-{v}" for k, v in colors.items()])
                if not color_str:
                    color_str = "nocolors"
                
                # Build filename
                filename = f"{timestamp}_det{i:02d}_{det.category.label}_conf{det.confidence:.2f}_{attr_str}_{color_str}.jpg"
                filepath = crops_dir / filename
                
                # Save crop
                crop.save(filepath, quality=95)
                logger.info(f"  💾 Saved crop {i}: {filepath.name} ({crop.width}x{crop.height})")
            
            logger.info(f"✅ Saved detection crops to {crops_dir}/")
            
        except Exception as e:
            logger.error(f"Failed to save detection crops: {e}", exc_info=True)
