"""Core MQTT client for device communication."""

import json
import logging
from typing import Optional, Dict, Any, Callable
import paho.mqtt.client as mqtt

logger = logging.getLogger(__name__)


class MQTTClient:
    """Thread-safe MQTT client with auto-reconnect and buffer-based I/O.
    
    Uses buffers for symmetric source/sink pattern:
    - Inbound: MQTT messages → topic_buffers → handler tasks
    - Outbound: publisher tasks → outbound_buffers → MQTT broker
    """
    
    def __init__(
        self,
        device_id: str,
        broker_url: str = "mqtt://localhost:1883",
        runner=None,
        validator=None,
        topic_buffers: Optional[Dict[str, Any]] = None,
        outbound_buffers: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize MQTT client.
        
        Args:
            device_id: Unique device identifier
            broker_url: MQTT broker URL (format: mqtt://host:port)
            runner: Optional runner for queuing tasks when messages arrive
            validator: Optional MQTTValidator for message validation
            topic_buffers: Dict mapping topic patterns to inbound Buffers
            outbound_buffers: Dict mapping message types to outbound Buffers
        """
        self.device_id = device_id
        self.broker_url = broker_url
        self.runner = runner
        self.validator = validator
        self.connected = False
        self._client: Optional[mqtt.Client] = None
        self._on_connect_callback: Optional[Callable] = None
        self._message_handlers: Dict[str, Callable] = {}
        
        # Buffer-based I/O
        self.topic_buffers = topic_buffers or {}  # topic -> Buffer
        self.outbound_buffers = outbound_buffers or {}  # msg_type -> Buffer
        
        # Parse broker URL
        mqtt_parts = broker_url.replace("mqtt://", "").split(":")
        self.host = mqtt_parts[0]
        self.port = int(mqtt_parts[1]) if len(mqtt_parts) > 1 else 1883
    
    def connect(self, on_connect: Optional[Callable] = None) -> bool:
        """
        Connect to MQTT broker.
        
        Args:
            on_connect: Optional callback when connected
            
        Returns:
            True if connected successfully
        """
        # Guard against duplicate connections
        if self._client is not None:
            logger.warning(f"Already have MQTT client for {self.device_id}, connected={self.connected}")
            if self.connected:
                logger.warning("Already connected, returning True")
                return True
            else:
                logger.error("Client exists but not connected - this should not happen!")
                return False
        
        try:
            logger.info(f"Connecting to MQTT broker at {self.host}:{self.port}")
            
            self._on_connect_callback = on_connect
            self._client = mqtt.Client(client_id=self.device_id)
            
            # Configure automatic reconnection with exponential backoff
            # Reconnect delay: 1s, 2s, 4s, 8s, 16s, 32s, up to 120s max
            self._client.reconnect_delay_set(min_delay=1, max_delay=120)
            
            # Setup callbacks
            self._client.on_connect = self._on_connect
            self._client.on_disconnect = self._on_disconnect
            self._client.on_message = self._on_message
            
            # Connect
            self._client.connect(self.host, self.port, keepalive=60)
            self._client.loop_start()
            
            # Wait for connection
            import time
            timeout = 5
            start = time.time()
            while not self.connected and (time.time() - start) < timeout:
                time.sleep(0.1)
            
            if not self.connected:
                logger.error("MQTT connection timeout")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"MQTT connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MQTT broker."""
        if self._client:
            self._client.loop_stop()
            self._client.disconnect()
            self.connected = False
    
    def subscribe(self, topic: str, handler: Callable[[Dict[str, Any]], None], qos: int = 1):
        """
        Subscribe to topic with message handler.
        
        Args:
            topic: MQTT topic to subscribe to
            handler: Callback function that receives parsed JSON payload
            qos: QoS level (0, 1, or 2)
        """
        if self._client:
            result = self._client.subscribe(topic, qos=qos)
            self._message_handlers[topic] = handler
            logger.info(f"Subscribed to: {topic} (result: {result})")
        else:
            logger.error(f"Cannot subscribe to {topic} - client not initialized")
    
    def publish(self, topic: str, payload: Dict[str, Any], qos: int = 0):
        """
        Publish message to topic.
        
        Args:
            topic: MQTT topic
            payload: Message payload (will be JSON encoded)
            qos: QoS level (0, 1, or 2)
        """
        if self._client and self.connected:
            msg_json = json.dumps(payload)
            result = self._client.publish(topic, msg_json, qos=qos)
            if result.rc != mqtt.MQTT_ERR_SUCCESS:
                logger.error(f"Failed to publish to {topic}")
        else:
            logger.warning(f"Cannot publish to {topic} - not connected")
    
    def _on_connect(self, client, userdata, flags, rc):
        """Internal: MQTT connect callback."""
        if rc == 0:
            logger.info("✓ Connected to MQTT broker")
            self.connected = True
            
            # Re-subscribe to all topics on reconnect
            if flags.get('session present', False):
                logger.info("Session persisted, subscriptions retained")
            else:
                logger.info("New session, re-subscribing to topics")
                for topic in self._message_handlers.keys():
                    logger.info(f"Re-subscribing to: {topic}")
                    self._client.subscribe(topic, qos=1)
            
            # Call connect callback (triggers device registration on initial connect)
            if self._on_connect_callback:
                self._on_connect_callback()
        else:
            logger.error(f"MQTT connection failed with code: {rc}")
            self.connected = False
    
    def _on_disconnect(self, client, userdata, rc):
        """Internal: MQTT disconnect callback."""
        self.connected = False
        if rc != 0:
            logger.error(f"❌ Unexpected disconnect from MQTT broker (code: {rc}, reason: {mqtt.error_string(rc)})")
            logger.error("Fatal: MQTT connection lost. Exiting to trigger full restart and re-registration.")
            import sys
            sys.exit(1)  # Exit to force full restart
        else:
            logger.info("Disconnected from MQTT broker (clean disconnect)")
    
    def _on_message(self, client, userdata, msg):
        """Internal: MQTT message callback - puts messages into topic buffers."""
        logger.info(f"📥📥📥 _on_message CALLBACK FIRED! Topic: {msg.topic}, Payload length: {len(msg.payload)}")
        
        try:
            payload = json.loads(msg.payload.decode())
            logger.info(f"📥 Message received on {msg.topic}, payload keys: {list(payload.keys())}")
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON from {msg.topic}: {e}")
            return
        
        logger.debug(f"Received message on {msg.topic}")
        
        # ============================================================
        # DO NOT REMOVE OR MODIFY - SECURITY CRITICAL
        # Validate all incoming MQTT messages before processing
        # ============================================================
        if self.validator:
            # This will terminate the device if validation fails
            self.validator.validate_message(msg.topic, payload)
            logger.info(f"✓ Message validated for {msg.topic}")
        # ============================================================
        # END SECURITY CRITICAL SECTION
        # ============================================================
        
        # Buffer-based flow: put message into topic buffer
        matched = False
        for topic_pattern, buffer in self.topic_buffers.items():
            if self._topic_matches(msg.topic, topic_pattern):
                logger.info(f"✓ Putting message into buffer for pattern: {topic_pattern}")
                # Create message dict with topic and payload
                message = {"topic": msg.topic, "payload": payload}
                buffer.put(message)
                matched = True
                break
        
        # Legacy handler support (for backward compatibility)
        if not matched:
            for topic_pattern, handler in self._message_handlers.items():
                if self._topic_matches(msg.topic, topic_pattern):
                    logger.info(f"✓ Handler found for {msg.topic} (pattern: {topic_pattern})")
                    handler(msg.topic, payload)
                    matched = True
                    break
        
        if not matched:
            logger.warning(f"⚠ No buffer or handler found for {msg.topic}")
    
    def _topic_matches(self, topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (supports wildcards)."""
        # Simple exact match for now, could add +/# wildcard support
        return topic == pattern or pattern in topic
