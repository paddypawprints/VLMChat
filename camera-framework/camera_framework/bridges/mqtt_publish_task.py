"""MQTT publish task that reads from outbound buffers and publishes to broker."""

import logging
from typing import Dict, Any
from ..task import BaseTask

logger = logging.getLogger(__name__)


class MQTTPublishTask(BaseTask):
    """Reads messages from outbound buffers and publishes to MQTT broker.
    
    This task bridges the buffer-based pipeline to MQTT publishing.
    It reads from its input buffers and publishes each message to MQTT.
    
    Expected message format in buffers:
    {
        "topic": "devices/{device_id}/logs",
        "payload": {...},  # Will be JSON-encoded
        "qos": 0  # Optional, defaults to 0
    }
    """
    
    def __init__(
        self,
        mqtt_client,
        name: str = "mqtt_publish",
        interval: float = 0.1,  # Check buffers every 100ms
    ):
        """Initialize MQTT publish task.
        
        Args:
            mqtt_client: MQTTClient instance for publishing
            name: Task name
            interval: How often to check buffers (seconds)
        """
        super().__init__(name=name, interval=interval)
        self.mqtt_client = mqtt_client
    
    def process(self) -> None:
        """Read messages from input buffers and publish to MQTT.
        
        Reads Context from input buffers, extracts MQTT messages,
        and publishes them to the broker.
        """
        # Read from input buffers
        if not self.inputs:
            return
        
        for buffer in self.inputs.values():  # Iterate over buffer values, not keys
            if not buffer.has_data():
                continue
                
            try:
                message_dict = buffer.get()
                
                # Skip if buffer returned None
                if message_dict is None:
                    continue
                
                # Process all messages from all fields in dict
                # message_dict is a plain dict with field_name -> list of messages
                for field_name, items in message_dict.items():
                    if not isinstance(items, list):
                        items = [items]
                    
                    for message in items:
                        if not isinstance(message, dict):
                            logger.warning(f"Skipping non-dict message: {type(message)}")
                            continue
                        
                        topic = message.get("topic")
                        payload = message.get("payload")
                        qos = message.get("qos", 0)
                        
                        if not topic or payload is None:
                            logger.warning(f"Message missing topic or payload: {message}")
                            continue
                        
                        # Publish to MQTT
                        try:
                            self.mqtt_client.publish(topic, payload, qos=qos)
                            logger.debug(f"Published to {topic}")
                        except Exception as e:
                            logger.error(f"Failed to publish to {topic}: {e}")
            except Exception as e:
                logger.error(f"Error processing buffer: {e}", exc_info=True)
