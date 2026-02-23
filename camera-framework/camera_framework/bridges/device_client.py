"""Simplified device client using runner task queue."""

import time
import logging
from datetime import datetime
from ..runner import Runner
from ..metrics import Collector
from .mqtt_client import MQTTClient
from ..mqtt_validator import MQTTValidator
from .registration import register_device
from .heartbeat import HeartbeatTask
from .metrics_task import MetricsPublishTask
from .log_task import LogPublishTask
from .snapshot_task import SnapshotTask

logger = logging.getLogger(__name__)


class DeviceClient:
    """Orchestrates MQTT device with runner-based task execution.
    
    MQTT thread handles network I/O, queues tasks to runner.
    Runner executes all business logic in main thread.
    """
    
    def __init__(
        self,
        device_id: str,
        device_type: str,
        runner: Runner,
        broker_host: str,
        broker_port: int = 1883,
        camera_source = None,
        yolo_buffer = None,
        schemas_path: str = None,
        detection_filter = None,  # Shared DetectionFilter instance
        sample_buffer = None,  # Buffer for snapshot to sample from
    ):
        self.device_id = device_id
        self.device_type = device_type
        self.runner = runner
        self.camera_source = camera_source
        self.yolo_buffer = yolo_buffer
        self.detection_filter = detection_filter  # Store filter for updates
        self.sample_buffer = sample_buffer  # Store for snapshot task
        
        # Initialize validator (will terminate device if schemas invalid)
        logger.info("Initializing MQTT message validator...")
        self.validator = MQTTValidator(device_id, schemas_path)
        
        # Metrics - use runner's collector or create new one
        self.collector = runner.collector
        if self.collector is None:
            from ..metrics import Collector
            self.collector = Collector()
            runner.collector = self.collector
        
        # Create MQTT client with runner and validator injected
        broker_url = f"mqtt://{broker_host}:{broker_port}"
        self.mqtt_client = MQTTClient(self.device_id, broker_url, runner=runner, validator=self.validator)
        
        # Create MQTT tasks (but don't subscribe yet - that happens after connection)
        self.heartbeat = HeartbeatTask(self.mqtt_client, self.device_id, interval=30.0)
        self.metrics = MetricsPublishTask(
            device_id=self.device_id,
            collector=self.collector,
            interval=5.0,  # Publish metrics every 5 seconds
        )
        self.metrics.runner = runner  # Give metrics access to runner for queueing config tasks
        self.logs = LogPublishTask(
            device_id=self.device_id,
            max_queue_size=1000,
        )
        
        # Create MQTT publish task for outbound messages
        from .mqtt_publish_task import MQTTPublishTask
        from ..buffer import Buffer, drop_oldest_policy
        
        self.mqtt_publish = MQTTPublishTask(
            mqtt_client=self.mqtt_client,
            name=f"mqtt_publish_{self.device_id}",
            interval=0.1
        )
        
        # Create buffer to connect metrics/logs to mqtt_publish
        outbound_buffer = Buffer(size=100, policy=drop_oldest_policy, name="outbound")
        
        # Connect tasks via buffers using new dict-based API
        self.metrics.add_output("messages", outbound_buffer)
        self.logs.add_output("messages", outbound_buffer)
        self.mqtt_publish.add_input("messages", outbound_buffer)
        
        # Add MQTT tasks to runner
        runner.tasks.extend([self.heartbeat, self.metrics, self.logs, self.mqtt_publish])
        
        # Create snapshot task if sample_buffer provided (add now, subscribe later in start())
        self.snapshot_task = None
        if sample_buffer is not None:
            logger.info(f"Creating snapshot task for {self.device_id} with buffer {sample_buffer.name}")
            from .snapshot_task import SnapshotTask
            self.snapshot_task = SnapshotTask(
                None,  # mqtt_client will be set before subscription in start()
                self.device_id,
                sample_buffer=sample_buffer,
                defer_subscription=True  # We'll subscribe in start()
            )
            self.snapshot_task.mqtt_client = self.mqtt_client  # Set reference
            self.snapshot_task.runner = runner
            runner.tasks.append(self.snapshot_task)
            logger.info(f"Snapshot task created and added to runner ({len(runner.tasks)} total tasks)")
    
    def start(self) -> None:
        """Connect MQTT and register device."""
        self.mqtt_client.connect()
        
        # Now that MQTT is connected, set up command subscriptions
        # Metrics command subscription (handles start/stop via enabled flag)
        self.mqtt_client.subscribe(
            f"devices/{self.device_id}/commands/metrics",
            self.metrics._handle_metrics_command,
        )
        
        # Log streaming command subscription
        self.mqtt_client.subscribe(
            f"devices/{self.device_id}/commands/logs",
            self._handle_log_command,
        )
        
        # Snapshot command subscription (task already created in __init__)
        if self.snapshot_task is not None:
            logger.info(f"Subscribing to snapshot command topic: devices/{self.device_id}/commands/snapshot")
            self.mqtt_client.subscribe(
                f"devices/{self.device_id}/commands/snapshot",
                self.snapshot_task._handle_snapshot_command,
            )
            logger.info(f"Snapshot command subscription active for {self.device_id}")
        
        # Filter command subscription
        if self.detection_filter:
            self.mqtt_client.subscribe(
                f"devices/{self.device_id}/commands/filter",
                self._handle_filter_command,
            )
            logger.info(f"Filter command subscription initialized for {self.device_id}")
        
        # Register device (one-time)
        register_device(self.mqtt_client, self.device_id, self.device_type)
        
        logger.info(f"Device {self.device_id} started")
    
    def _handle_log_command(self, topic: str, payload: dict) -> None:
        """MQTT callback - queue log config task to runner."""
        from .log_task import LogConfigTask
        if self.mqtt_client.runner:
            config_task = LogConfigTask(payload, self.logs)
            self.mqtt_client.runner.queue_task(config_task)
        else:
            logger.warning("MQTT client runner not set, cannot queue log config task")
    
    def _handle_filter_command(self, topic: str, payload: dict) -> None:
        """MQTT callback - queue filter config task to runner."""
        from .filter_task import FilterConfigTask
        
        logger.info(f"[DeviceClient] Received filter command on topic: {topic}")
        logger.info(f"[DeviceClient] Payload keys: {list(payload.keys())}")
        logger.info(f"[DeviceClient] Payload: {payload}")
        
        if self.mqtt_client.runner and self.detection_filter:
            config_task = FilterConfigTask(payload, self.detection_filter)
            self.mqtt_client.runner.queue_task(config_task)
            
            # Log filter count if present
            filters = payload.get('filters', [])
            logger.info(f"[DeviceClient] Queued filter update with {len(filters)} filters")
            for f in filters:
                logger.info(f"[DeviceClient]   - {f.get('name', 'unknown')}: cats={sum(f.get('category_mask', []))}/80, attrs={sum(f.get('attribute_mask', []))}/26")
        else:
            logger.warning("MQTT client runner or detection_filter not set, cannot queue filter config task")
    
    def stop(self) -> None:
        """Cleanup and disconnect."""
        # Send offline status
        topic = f"devices/{self.device_id}/status"
        message = {
            "device_id": self.device_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "offline",
        }
        self.mqtt_client.publish(topic, message)
        
        # Disconnect MQTT
        self.mqtt_client.disconnect()
        
        logger.info(f"Device {self.device_id} stopped")
