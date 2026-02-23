"""VLM Queue - Smart routing for VLM verification."""

import logging
import time
from typing import Optional, Dict, Any
from camera_framework import BaseTask


logger = logging.getLogger(__name__)


class VLMQueue(BaseTask):
    """
    Smart router for VLM verification pipeline.
    
    Routing logic:
    1. If vlm_required=False → Route to "alerts" output immediately
    2. If vlm_required=True AND SmolVLM ready → Route to "vlm" output
    3. If vlm_required=True AND SmolVLM busy → Queue message
    4. If queue full → Drop oldest, send to "alerts" with vlm_timeout=True
    
    Named Outputs:
        "alerts": Alert publisher (immediate alerts + timeouts)
        "vlm": SmolVLM worker (for VLM processing)
    
    Example:
        vlm_queue = VLMQueue(max_queue_size=10, smolvlm_worker=vlm_worker)
        
        # Wire: tracker → vlm_queue → {alerts, vlm}
        tracker_buffer = Buffer(size=30, policy=drop_oldest_policy)
        alert_buffer = Buffer(size=30, policy=drop_oldest_policy)
        vlm_buffer = Buffer(size=1, policy=blocking_policy)
        
        tracker.add_output("out", tracker_buffer)
        vlm_queue.add_input("in", tracker_buffer)
        vlm_queue.add_output("alerts", alert_buffer)
        vlm_queue.add_output("vlm", vlm_buffer)
    """
    
    def __init__(
        self,
        max_queue_size: int = 10,
        smolvlm_worker: Optional['SmolVLMVerifier'] = None,
        name: str = "vlm_queue",
    ):
        """
        Initialize VLM queue router.
        
        Args:
            max_queue_size: Maximum queued detections before timeout
            smolvlm_worker: Reference to SmolVLM worker for busy check
            name: Task name
        """
        super().__init__(name=name)
        self.max_queue_size = max_queue_size
        self.smolvlm_worker = smolvlm_worker
        
        # Queue for detections waiting for VLM
        self.pending_queue = []
        
        # Metrics
        self.stats = {
            'routed_to_alerts': 0,
            'routed_to_vlm': 0,
            'queued': 0,
            'timeouts': 0,
            'queue_full_drops': 0,
        }
    
    def set_smolvlm_worker(self, worker: 'SmolVLMVerifier') -> None:
        """Set SmolVLM worker reference (for dependency injection)."""
        self.smolvlm_worker = worker
    
    def process(self) -> None:
        """Route detection based on vlm_required flag and SmolVLM availability."""
        # Process any queued messages first if SmolVLM is ready
        self._process_queued_messages()
        
        # Get new detection from input
        input_buffer = list(self.inputs.values())[0]
        message = input_buffer.get()
        if not message:
            return
        
        # Extract detection event from message dict
        # Assuming DetectionTracker puts confirmation events in "confirmations" field
        confirmations = message.get("confirmations", [])
        if not confirmations:
            logger.debug("No confirmations in message, skipping")
            return
        
        for event in confirmations:
            if not isinstance(event, dict):
                logger.warning(f"Received non-dict event: {type(event)}")
                continue
            
            # Check vlm_required flag
            vlm_required = event.get("vlm_required", False)
            
            if not vlm_required:
                # Route directly to alerts (no VLM needed)
                self._route_to_alerts(event)
                self.stats['routed_to_alerts'] += 1
                logger.debug(f"Routed track {event.get('track_id')} to alerts (no VLM required)")
                continue
            
            # VLM verification required - check if worker is available
            if self._is_smolvlm_ready():
                # SmolVLM is ready - route directly
                self._route_to_smolvlm(event)
                self.stats['routed_to_vlm'] += 1
                logger.info(f"Routed track {event.get('track_id')} to SmolVLM (ready)")
            else:
                # SmolVLM is busy or not configured - queue the message
                if len(self.pending_queue) >= self.max_queue_size:
                    # Queue is full - drop oldest and send to alerts with timeout
                    oldest = self.pending_queue.pop(0)
                    self._route_to_alerts(oldest, vlm_timeout=True)
                    self.stats['timeouts'] += 1
                    self.stats['queue_full_drops'] += 1
                    logger.warning(
                        f"⏱️  VLM queue full ({self.max_queue_size}), timed out track {oldest.get('track_id')} "
                        f"(queued for {time.time() - oldest.get('queued_at', 0):.1f}s) - sending as YELLOW alert"
                    )
                
                # Add to queue with timestamp
                event['queued_at'] = time.time()
                self.pending_queue.append(event)
                self.stats['queued'] += 1
                logger.info(
                    f"📥 Queued track {event.get('track_id')} for VLM verification "
                    f"(queue: {len(self.pending_queue)}/{self.max_queue_size}, worker_ready={self._is_smolvlm_ready()})"
                )
    
    def _process_queued_messages(self) -> None:
        """Process queued messages if SmolVLM is ready."""
        if not self.pending_queue:
            return
        
        if not self._is_smolvlm_ready():
            return
        
        # SmolVLM is ready - send next queued message
        event = self.pending_queue.pop(0)
        queue_time = time.time() - event.get('queued_at', 0)
        
        self._route_to_smolvlm(event)
        self.stats['routed_to_vlm'] += 1
        logger.info(
            f"Routed queued track {event.get('track_id')} to SmolVLM "
            f"(waited {queue_time:.1f}s, queue: {len(self.pending_queue)})"
        )
    
    def _is_smolvlm_ready(self) -> bool:
        """Check if SmolVLM worker is ready to accept new work.
        
        Returns:
            True if SmolVLM can process, False if busy or unavailable
        """
        if not self.smolvlm_worker:
            # No worker configured - route to alerts
            return False
        
        # Check worker's busy state
        return not self.smolvlm_worker.is_busy()
    
    def _route_to_alerts(self, event: Dict[str, Any], vlm_timeout: bool = False) -> None:
        """Send detection to alert publisher.
        
        Args:
            event: Detection event
            vlm_timeout: If True, mark as VLM timeout
        """
        if vlm_timeout:
            event['vlm_status'] = 'timeout'
            event['vlm_confirmed'] = False
            event['vlm_timeout'] = True  # legacy field
        
        # Remove queuing metadata if present
        event.pop('queued_at', None)
        
        # Create context with event and send to "alerts" output
        message = {}
        message.setdefault("confirmations", []).append(event)
        
        if "alerts" in self.outputs:
            self.outputs["alerts"].put(message)
    
    def _route_to_smolvlm(self, event: Dict[str, Any]) -> None:
        """Send detection to SmolVLM worker.
        
        Args:
            event: Detection event
        """
        # Remove queuing metadata
        event.pop('queued_at', None)
        
        # Create context with event and send to "vlm" output
        message = {}
        message.setdefault("confirmations", []).append(event)
        
        if "vlm" in self.outputs:
            self.outputs["vlm"].put(message)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics.
        
        Returns:
            Dict with routing stats and queue status
        """
        return {
            **self.stats,
            'queue_size': len(self.pending_queue),
            'queue_capacity': self.max_queue_size,
            'smolvlm_ready': self._is_smolvlm_ready(),
        }
