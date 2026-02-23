"""Metrics publishing to MQTT."""

import logging
import threading
import time
from datetime import datetime
from typing import Optional, Dict, Any
from camera_framework import Collector
from .mqtt_client import MQTTClient

logger = logging.getLogger(__name__)


class MetricsPublisher:
    """Publishes metrics to MQTT on schedule."""
    
    def __init__(self, device_id: str, collector: Collector, mqtt_client: MQTTClient):
        """
        Initialize metrics publisher.
        
        Args:
            device_id: Unique device identifier
            collector: Metrics collector instance
            mqtt_client: MQTT client instance
        """
        self.device_id = device_id
        self.collector = collector
        self.mqtt = mqtt_client
        
        self.enabled = False
        self.frequency = 30  # seconds
        self.instruments_filter: Optional[list] = None
        
        self._thread: Optional[threading.Thread] = None
        self._running = False
        
        # Subscribe to metrics commands
        command_topic = f"devices/{device_id}/commands/metrics"
        mqtt_client.subscribe(command_topic, self._handle_command, qos=1)
    
    def _handle_command(self, payload: Dict[str, Any]):
        """
        Handle metrics configuration command.
        
        Payload format:
        {
            "enabled": true,
            "frequency": 30,
            "instruments": ["pipeline.fps", "pipeline.duration"] or "*"
        }
        """
        try:
            enabled = payload.get('enabled', False)
            frequency = payload.get('frequency', self.frequency)
            instruments = payload.get('instruments')
            
            # Enforce minimum frequency
            MIN_FREQUENCY = 10
            if frequency < MIN_FREQUENCY:
                logger.warning(f"Frequency {frequency}s too low, using {MIN_FREQUENCY}s")
                frequency = MIN_FREQUENCY
            
            logger.info(f"Metrics command: enabled={enabled}, frequency={frequency}s")
            
            self.frequency = frequency
            
            if instruments == "*" or instruments is None:
                self.instruments_filter = None
            elif isinstance(instruments, list):
                self.instruments_filter = instruments[:50]  # Max 50 instruments
            
            if enabled:
                self.start()
            else:
                self.stop()
                
        except Exception as e:
            logger.error(f"Error handling metrics command: {e}")
    
    def start(self):
        """Start metrics publishing thread."""
        if self._thread and self._thread.is_alive():
            logger.info("Metrics publishing already running")
            return
        
        self.enabled = True
        self._running = True
        self._thread = threading.Thread(target=self._publish_loop, daemon=True)
        self._thread.start()
        logger.info(f"✓ Metrics publishing started (every {self.frequency}s)")
    
    def stop(self):
        """Stop metrics publishing thread."""
        self.enabled = False
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("✓ Metrics publishing stopped")
    
    def _publish_loop(self):
        """Background loop that publishes metrics."""
        while self._running:
            if self.enabled:
                try:
                    self._publish_metrics()
                except Exception as e:
                    logger.error(f"Error publishing metrics: {e}")
            
            time.sleep(self.frequency)
    
    def _publish_metrics(self):
        """Publish current metrics to MQTT."""
        try:
            # Get all instrument stats
            all_stats = self.collector.get_all_stats()
            
            # Apply filter if set
            if self.instruments_filter:
                all_stats = {
                    k: v for k, v in all_stats.items()
                    if k in self.instruments_filter
                }
            
            metrics_message = {
                "timestamp": datetime.now().isoformat(),
                "metrics": all_stats
            }
            
            topic = f"devices/{self.device_id}/metrics"
            self.mqtt.publish(topic, metrics_message, qos=0)
            logger.debug(f"Published {len(all_stats)} metrics")
            
        except Exception as e:
            logger.error(f"Failed to publish metrics: {e}")
