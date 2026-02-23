"""Metrics publishing task."""

import time
import logging
from typing import Optional, List
from ..task import BaseTask
from ..metrics import Collector

logger = logging.getLogger(__name__)


class MetricsConfigTask(BaseTask):
    """One-off task to handle metrics configuration from MQTT."""
    
    def __init__(self, payload: dict, metrics_publish_task):
        super().__init__(name="metrics_config", interval=None)
        self.payload = payload
        self.metrics_publish_task = metrics_publish_task
    
    def process(self) -> None:
        """Apply metrics configuration."""
        self.metrics_publish_task.enabled = self.payload.get("enabled", True)
        
        logger.info(f"[MetricsConfig] Applying config: {self.payload}")
        logger.info(f"[MetricsConfig] Metrics enabled: {self.metrics_publish_task.enabled}")
        
        if "frequency" in self.payload:
            # Update interval (minimum 10s)
            new_interval = max(10.0, float(self.payload["frequency"]))
            self.metrics_publish_task.interval = new_interval
            self.metrics_publish_task.last_run = 0  # Reset to trigger immediate publish
        
        if "instruments_filter" in self.payload:
            self.metrics_publish_task.instruments_filter = self.payload["instruments_filter"]
        
        logger.info(f"Metrics config updated: enabled={self.metrics_publish_task.enabled}, "
                   f"interval={self.metrics_publish_task.interval}s, "
                   f"filter={self.metrics_publish_task.instruments_filter}")


class MetricsPublishTask(BaseTask):
    """Periodic metrics publishing task that writes to outbound buffer.
    
    Publishes metrics via outbound buffer → MQTTPublishTask at configurable interval.
    Supports filtering to only publish specific instruments.
    """
    
    def __init__(
        self,
        device_id: str,
        collector: Collector,
        interval: float = 60.0,
        instruments_filter: Optional[List[str]] = None,
    ):
        super().__init__(name="metrics_publish", interval=interval)
        self.device_id = device_id
        self.collector = collector
        self.instruments_filter = instruments_filter
        self.enabled = False  # Start disabled, enabled via metrics_start command
        
        # Note: Subscription to commands/metrics is now done in DeviceClient.start()
        # after MQTT connection is established
    
    def _handle_metrics_command(self, topic: str, payload: dict) -> None:
        """MQTT callback - queue config task to runner.
        
        Args:
            topic: MQTT topic
            payload: Command payload with 'enabled', 'frequency', 'instruments_filter'
        """
        logger.info(f"[MetricsPublish] Metrics command received: {payload}")
        
        # Queue config task to runner
        if self.runner:
            config_task = MetricsConfigTask(payload, self)
            self.runner.queue_task(config_task)
            logger.info(f"[MetricsPublish] Queued config task to runner")
        else:
            logger.error("No runner available for metrics config task!")
    
    def is_ready(self) -> bool:
        """Always ready when enabled (source task with no inputs)."""
        return self.enabled
    
    def process(self) -> None:
        """Collect metrics and write to Context (flows to outbound buffer)."""
        if not self.enabled:
            logger.debug(f"[MetricsPublish] Skipping - disabled")
            return
        
        logger.info(f"[MetricsPublish] Publishing metrics...")
        
        # Get all stats from collector
        stats = self.collector.get_all_stats()
        
        logger.info(f"[MetricsPublish] Collector has {len(stats)} stats: {list(stats.keys())}")
        
        # Apply filter if configured
        if self.instruments_filter:
            stats = {k: v for k, v in stats.items() if k in self.instruments_filter}
        
        if not stats:
            logger.warning(f"[MetricsPublish] No stats to publish (filter={self.instruments_filter})")
            return
        
        # Convert stats to schema format
        from datetime import datetime, timezone
        instruments = []
        for name, stat_data in stats.items():
            # Special handling for memory instrument
            if "total_alive" in stat_data:
                # Memory instrument - add multiple metrics
                instruments.append({
                    "name": f"{name}.alive",
                    "type": "counter",
                    "value": stat_data["total_alive"]
                })
                instruments.append({
                    "name": f"{name}.cleaned",
                    "type": "counter", 
                    "value": stat_data["total_cleaned"]
                })
                # Add total leaked count
                if stat_data.get("potential_leaks"):
                    total_leaked = sum(leak['count'] for leak in stat_data['potential_leaks'])
                    instruments.append({
                        "name": f"{name}.leaked",
                        "type": "counter",
                        "value": total_leaked
                    })
            # Determine instrument type and value
            elif "rate" in stat_data:
                instruments.append({
                    "name": name,
                    "type": "rate",
                    "value": stat_data["rate"]
                })
            elif "avg" in stat_data:
                instruments.append({
                    "name": name,
                    "type": "avg_duration",
                    "value": stat_data["avg"]
                })
            elif "count" in stat_data:
                instruments.append({
                    "name": name,
                    "type": "counter",
                    "value": stat_data["count"]
                })
        
        if not instruments:
            return
        
        # Create message dict (read from buffer if available, otherwise create new)
        message = {}
        if self.inputs:
            buffer = list(self.inputs.values())[0]
            if buffer.has_data():
                message = buffer.get()
        
        # Create MQTT message for outbound buffer
        topic = f"devices/{self.device_id}/metrics"
        message_payload = {
            "session": f"{self.device_id}-session",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "instruments": instruments,
        }
        
        mqtt_message = {
            "topic": topic,
            "payload": message_payload,
            "qos": 0
        }
        
        # Add to message
        output_field = self.field("mqtt_message")
        message.setdefault(output_field, []).append(mqtt_message)
        
        # Write to output buffers
        if self.outputs:
            for buffer in self.outputs.values():
                buffer.put(message)
