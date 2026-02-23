"""Device client orchestrator."""

import logging
from typing import Optional, List
from camera_framework import Runner, Collector, BaseTask
from .mqtt_client import MQTTClient
from .registration import Registration
from .metrics_publisher import MetricsPublisher

logger = logging.getLogger(__name__)


class DeviceClient:
    """
    Orchestrates device registration, MQTT communication, and pipeline execution.
    
    Usage:
        client = DeviceClient(device_id="mac-01", device_type="mac")
        client.add_tasks([Camera(), YOLO(), Display()])
        client.run()
    """
    
    def __init__(self,
                 device_id: str,
                 device_type: str = "mac",
                 mqtt_url: str = "mqtt://localhost:1883",
                 device_name: Optional[str] = None):
        """
        Initialize device client.
        
        Args:
            device_id: Unique device identifier
            device_type: Device type (mac, jetson, pi)
            mqtt_url: MQTT broker URL
            device_name: Human-readable device name (optional)
        """
        self.device_id = device_id
        self.device_type = device_type
        self.device_name = device_name
        
        # Core components
        self.mqtt = MQTTClient(device_id, mqtt_url)
        self.registration = Registration(device_id, device_type, self.mqtt, device_name)
        self.collector = Collector(f"device_{device_id}")
        self.metrics_publisher = MetricsPublisher(device_id, self.collector, self.mqtt)
        
        # Pipeline
        self.runner: Optional[Runner] = None
        self._tasks: List[BaseTask] = []
        self._running = False
        
        logger.info(f"Device client initialized: {device_id}")
    
    def add_tasks(self, tasks: List[BaseTask]):
        """
        Add tasks to the pipeline.
        
        Args:
            tasks: List of BaseTask instances
        """
        self._tasks = tasks
        self.runner = Runner(tasks)
        logger.info(f"Added {len(tasks)} tasks to pipeline")
    
    def start(self) -> bool:
        """
        Start device client (connect MQTT, register, start heartbeat).
        
        Returns:
            True if started successfully
        """
        logger.info("Starting device client...")
        
        # Connect to MQTT
        if not self.mqtt.connect(on_connect=self._on_mqtt_connected):
            logger.error("Failed to connect to MQTT")
            return False
        
        self._running = True
        logger.info("✓ Device client started")
        return True
    
    def stop(self):
        """Stop device client (send offline status, disconnect)."""
        logger.info("Stopping device client...")
        
        self._running = False
        
        # Send offline status
        from datetime import datetime
        if self.mqtt.connected:
            status = {
                "status": "offline",
                "timestamp": datetime.now().isoformat(),
            }
            topic = f"devices/{self.device_id}/status"
            self.mqtt.publish(topic, status, qos=1)
        
        # Stop components
        self.metrics_publisher.stop()
        self.registration.stop_heartbeat()
        self.mqtt.disconnect()
        
        logger.info("✓ Device client stopped")
    
    def run(self):
        """
        Main run loop - start client and run pipeline continuously.
        
        Blocks until KeyboardInterrupt.
        """
        if not self.runner:
            logger.error("No tasks added - call add_tasks() first")
            return
        
        if not self.start():
            return
        
        try:
            logger.info("Device running (Ctrl+C to exit)...")
            
            while self._running:
                # Run pipeline once
                self.runner.run_once()
                
                # Record frame for FPS tracking
                self.collector.record("pipeline.frame", 1.0)
                
        except KeyboardInterrupt:
            logger.info("\nShutdown requested...")
        finally:
            self.stop()
    
    def _on_mqtt_connected(self):
        """Called when MQTT connection established."""
        # Register device
        instruments = self._get_available_instruments()
        self.registration.register(instruments=instruments)
        
        # Start heartbeat
        self.registration.start_heartbeat()
        
        logger.info("✓ Device registered and heartbeat started")
    
    def _get_available_instruments(self) -> list:
        """Get list of available metric instruments."""
        # For now, return empty list - instruments are added dynamically
        # Could introspect self.collector to get registered instruments
        return []
