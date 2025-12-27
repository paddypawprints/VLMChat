"""
Stream sources for event-driven pipeline execution.

Sources continuously produce data (camera frames, MQTT messages, etc.) 
that can be consumed by pipelines using wait() and latest() tasks.
"""

import logging
import threading
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)


class StreamSource(ABC):
    """
    Base class for stream sources that produce data for pipelines.
    
    Sources run independently and maintain their own data buffers.
    Pipelines consume data using wait() and latest() tasks.
    
    Design:
    - Sources DO NOT inherit from BaseTask (different lifecycle)
    - Sources are polled by PipelineRunner's background thread
    - Sources maintain ring buffers and track cursor positions
    """
    
    def __init__(self, name: str, buffer_size: int = 300, output_label: str = "default"):
        """
        Initialize stream source.
        
        Args:
            name: Unique identifier for this source
            buffer_size: Number of items to keep in ring buffer
            output_label: Label to use when adding data to context (e.g., "frame", "sensor_data")
        """
        self.name = name
        self.output_label = output_label
        self.buffer_size = buffer_size
        self.ring_buffer = deque(maxlen=buffer_size)
        self.sequence_counter = 0
        self.running = False
        self.lock = threading.Lock()
        
        # Track waiting cursors for wake-up
        self.waiting_cursors = {}  # cursor_id -> cursor
        
        # Track last consumed sequence per cursor (for next())
        self.cursor_positions = {}  # cursor_id -> last_sequence
        
        # Latest data for quick access
        self.latest_data = None
        self.latest_sequence = -1
    
    @abstractmethod
    def _capture_data(self) -> Optional[Any]:
        """
        Capture/receive new data from source.
        
        Called by poll() - implement source-specific logic here.
        Should be non-blocking or very fast.
        
        Returns:
            New data item, or None if no data available
        """
        pass
    
    def poll(self) -> bool:
        """
        Poll source for new data. Called by PipelineRunner's polling thread.
        
        Returns:
            True if new data was captured, False otherwise
        """
        if not self.running:
            return False
        
        try:
            new_data = self._capture_data()
            
            if new_data is not None:
                with self.lock:
                    # Add to ring buffer with sequence number
                    self.ring_buffer.append((self.sequence_counter, new_data))
                    self.latest_data = new_data
                    self.latest_sequence = self.sequence_counter
                    self.sequence_counter += 1
                
                logger.debug(f"Source '{self.name}' captured data #{self.latest_sequence}")
                return True
        except Exception as e:
            logger.error(f"Source '{self.name}' poll error: {e}")
        
        return False
    
    def has_new_data(self, cursor_id: Optional[int] = None) -> bool:
        """
        Check if new data is available.
        
        Args:
            cursor_id: Optional cursor ID to check for new data since last consumption
            
        Returns:
            True if new data available
        """
        with self.lock:
            if cursor_id is None:
                # Generic check - any data available?
                return len(self.ring_buffer) > 0
            
            # Check if cursor has unconsumed data
            last_consumed = self.cursor_positions.get(cursor_id, -1)
            return self.latest_sequence > last_consumed
    
    def get_latest(self, cursor_id: Optional[int] = None) -> Tuple[Optional[Any], int]:
        """
        Get the most recent data item, skipping any old items.
        
        Args:
            cursor_id: Optional cursor ID to track position
            
        Returns:
            Tuple of (data, sequence_number) or (None, -1) if no data
        """
        with self.lock:
            if self.latest_data is None:
                return None, -1
            
            if cursor_id is not None:
                self.cursor_positions[cursor_id] = self.latest_sequence
            
            return self.latest_data, self.latest_sequence
    
    def get_next(self, cursor_id: int) -> Tuple[Optional[Any], int]:
        """
        Get the next sequential data item for this cursor.
        
        Args:
            cursor_id: Cursor ID to track sequential position
            
        Returns:
            Tuple of (data, sequence_number) or (None, -1) if not ready
        """
        with self.lock:
            expected_sequence = self.cursor_positions.get(cursor_id, 0)
            
            # Search ring buffer for expected sequence
            for seq, data in self.ring_buffer:
                if seq == expected_sequence:
                    self.cursor_positions[cursor_id] = expected_sequence + 1
                    return data, seq
            
            # Not found - either too old (dropped from buffer) or not yet available
            if expected_sequence < self.sequence_counter - len(self.ring_buffer):
                # Too old - data was dropped
                logger.warning(f"Source '{self.name}': cursor {cursor_id} missed sequence {expected_sequence}")
                # Skip to oldest available
                if self.ring_buffer:
                    oldest_seq, oldest_data = self.ring_buffer[0]
                    self.cursor_positions[cursor_id] = oldest_seq + 1
                    return oldest_data, oldest_seq
            
            # Not yet available
            return None, -1
    
    def register_waiting_cursor(self, cursor_id: int, cursor: Any) -> None:
        """
        Register a cursor as waiting for data from this source.
        
        Args:
            cursor_id: Cursor ID
            cursor: Cursor object to wake when data arrives
        """
        with self.lock:
            self.waiting_cursors[cursor_id] = cursor
            logger.debug(f"Source '{self.name}': cursor {cursor_id} waiting")
    
    def unregister_waiting_cursor(self, cursor_id: int) -> None:
        """
        Remove cursor from waiting list.
        
        Args:
            cursor_id: Cursor ID to remove
        """
        with self.lock:
            self.waiting_cursors.pop(cursor_id, None)
    
    def get_waiting_cursors(self) -> Dict[int, Any]:
        """
        Get all cursors currently waiting for data.
        
        Returns:
            Dict of cursor_id -> cursor
        """
        with self.lock:
            return dict(self.waiting_cursors)
    
    def start(self) -> None:
        """Start the source."""
        self.running = True
        logger.info(f"Source '{self.name}' started")
    
    def stop(self) -> None:
        """Stop the source."""
        self.running = False
        logger.info(f"Source '{self.name}' stopped")
    
    def reset_cursor_position(self, cursor_id: int) -> None:
        """
        Reset cursor position to latest.
        
        Args:
            cursor_id: Cursor ID to reset
        """
        with self.lock:
            self.cursor_positions[cursor_id] = self.latest_sequence
