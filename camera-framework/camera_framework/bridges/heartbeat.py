"""Heartbeat task for device health monitoring."""

import time
from datetime import datetime
from ..task import BaseTask


class HeartbeatTask(BaseTask):
    """Periodic heartbeat task.
    
    Publishes heartbeat every 30 seconds to indicate device is alive.
    """
    
    def __init__(self, mqtt_client, device_id: str, interval: float = 30.0):
        super().__init__(name="heartbeat", interval=interval)
        self.mqtt_client = mqtt_client
        self.device_id = device_id
    
    def process(self) -> None:
        """Send heartbeat message."""
        topic = f"devices/{self.device_id}/heartbeat"
        message = {
            "device_id": self.device_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "status": "online",
        }
        self.mqtt_client.publish(topic, message)
