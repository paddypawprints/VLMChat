"""Log streaming task for real-time log delivery via MQTT."""

import logging
import queue
import threading
from datetime import datetime, timezone
from typing import Optional
from ..task import BaseTask

logger = logging.getLogger(__name__)


class MQTTLogHandler(logging.Handler):
    """Custom log handler that queues log records for MQTT publishing.
    
    This handler is thread-safe and non-blocking - it queues records
    instead of publishing directly to avoid blocking the logger.
    """
    
    def __init__(self, log_queue: queue.Queue, level_filter: str = "WARNING"):
        super().__init__()
        self.log_queue = log_queue
        self.level_filter = getattr(logging, level_filter, logging.WARNING)
        self.setLevel(self.level_filter)
    
    def emit(self, record: logging.LogRecord) -> None:
        """Queue log record for publishing."""
        try:
            # Only queue if level meets filter
            if record.levelno >= self.level_filter:
                self.log_queue.put_nowait(record)
        except queue.Full:
            # Drop logs if queue is full (backpressure)
            pass
        except Exception:
            # Don't let handler errors break logging
            self.handleError(record)
    
    def update_level(self, level: str) -> None:
        """Update the minimum log level filter."""
        self.level_filter = getattr(logging, level, logging.WARNING)
        self.setLevel(self.level_filter)


class LogConfigTask(BaseTask):
    """One-off task to handle log streaming configuration from MQTT."""
    
    def __init__(self, payload: dict, log_publish_task):
        super().__init__(name="log_config", interval=None)
        self.payload = payload
        self.log_publish_task = log_publish_task
    
    def process(self) -> None:
        """Apply log streaming configuration."""
        enabled = self.payload.get("enabled", False)
        level = self.payload.get("level", "WARNING")
        
        logger.info(f"[LogConfig] Received payload: {self.payload}")
        logger.info(f"[LogConfig] Setting enabled={enabled}, level={level}")
        
        self.log_publish_task.set_enabled(enabled, level)
        
        logger.info(f"[LogConfig] Log streaming config applied: enabled={enabled}, level={level}")


class LogPublishTask(BaseTask):
    """Log streaming task that writes log entries to outbound buffer.
    
    Uses a custom log handler to capture logs from the Python logging system
    and writes them to Context (flows to outbound buffer → MQTTPublishTask).
    
    Features:
    - Thread-safe log capture via queue
    - Configurable log level filtering
    - Non-blocking log handler (won't block application logging)
    - Automatic cleanup on disable
    - Backpressure handling (drops logs if queue full)
    """
    
    def __init__(
        self,
        device_id: str,
        max_queue_size: int = 1000,
    ):
        """Initialize log streaming task.
        
        Args:
            device_id: Device identifier
            max_queue_size: Maximum queued log records (backpressure limit)
        """
        super().__init__(name="log_publish", interval=0.1)  # Check queue every 100ms
        self.device_id = device_id
        self.enabled = False
        self.current_level = "WARNING"
        
        # Thread-safe queue for log records
        self.log_queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        
        # Custom log handler
        self.log_handler: Optional[MQTTLogHandler] = None
        self._lock = threading.Lock()
    
    def set_enabled(self, enabled: bool, level: str = "WARNING") -> None:
        """Enable or disable log streaming.
        
        Args:
            enabled: True to start streaming, False to stop
            level: Minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        logger.info(f"[LogPublishTask] set_enabled called: enabled={enabled}, level={level}, current_enabled={self.enabled}")
        
        with self._lock:
            if enabled and not self.enabled:
                # Start streaming
                logger.info(f"[LogPublishTask] Starting streaming at {level} level")
                self._start_streaming(level)
            elif not enabled and self.enabled:
                # Stop streaming
                logger.info("[LogPublishTask] Stopping streaming")
                self._stop_streaming()
            elif enabled and self.current_level != level:
                # Update level
                logger.info(f"[LogPublishTask] Updating level from {self.current_level} to {level}")
                if self.log_handler:
                    self.log_handler.update_level(level)
                    self.current_level = level
                    logger.info(f"Log level updated to {level}")
    
    def _start_streaming(self, level: str) -> None:
        """Attach log handler to root logger."""
        # Create handler
        self.log_handler = MQTTLogHandler(self.log_queue, level)
        self.current_level = level
        
        # Attach to root logger to capture all logs
        root_logger = logging.getLogger()
        root_logger.addHandler(self.log_handler)
        
        self.enabled = True
        logger.info(f"Log streaming started at {level} level")
    
    def _stop_streaming(self) -> None:
        """Detach log handler and clear queue."""
        if self.log_handler:
            # Remove handler from root logger
            root_logger = logging.getLogger()
            root_logger.removeHandler(self.log_handler)
            self.log_handler = None
        
        # Clear queue
        while not self.log_queue.empty():
            try:
                self.log_queue.get_nowait()
            except queue.Empty:
                break
        
        self.enabled = False
        logger.info("Log streaming stopped")
    
    def process(self) -> None:
        """Process queued log records and write to outbound buffer."""
        if not self.enabled:
            return
        
        # Read/create dict from buffers (if connected to runner)
        ctx = {}
        if self.inputs:
            # Read from first input buffer if available
            buffer = list(self.inputs.values())[0]
            if buffer.has_data():
                ctx = buffer.get()
        
        # Process up to 10 log records per tick to avoid blocking
        batch_size = 0
        max_batch = 10
        
        while batch_size < max_batch:
            try:
                record = self.log_queue.get_nowait()
                self._add_log_to_context(record, ctx)
                batch_size += 1
            except queue.Empty:
                break
        
        # Write to all output buffers (if connected)
        if self.outputs:
            for buffer in self.outputs.values():  # Iterate over buffer values, not keys
                buffer.put(ctx)
    
    def _add_log_to_context(self, record: logging.LogRecord, message: dict) -> None:
        """Add log record to message dict as MQTT message.
        
        Message will be written to outbound buffer by Runner,
        then published by MQTTPublishTask.
        
        Args:
            record: Python logging.LogRecord
            message: Dict to add MQTT message to
        """
        # Convert log record to schema format
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add optional fields if available
        if record.pathname:
            log_entry["module"] = record.filename
        if record.lineno:
            log_entry["line"] = record.lineno
        if record.threadName:
            log_entry["thread"] = record.threadName
        
        # Create MQTT message for outbound buffer
        topic = f"devices/{self.device_id}/logs"
        mqtt_message = {
            "topic": topic,
            "payload": log_entry,
            "qos": 0
        }
        
        # Add to message dict (Runner will write to output buffer)
        output_field = self.field("mqtt_message")
        if output_field not in message:
            message[output_field] = []
        message[output_field].append(mqtt_message)
    
    def cleanup(self) -> None:
        """Clean up resources on shutdown."""
        self._stop_streaming()
        super().cleanup()
