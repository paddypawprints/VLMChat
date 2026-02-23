"""SmolVLM verification task — yes/no confirmation of detection crops."""

import logging
import threading
import time
from pathlib import Path
from typing import Optional, Any, Dict, Callable
from PIL import Image
import io

from camera_framework import BaseTask
from .models.smolvlm_onnx import SmolVLMOnnx

logger = logging.getLogger(__name__)


class SmolVLMVerifier(BaseTask):
    """
    Background VLM verification worker.
    
    Processes detections asynchronously without blocking the main pipeline.
    Signals busy/ready state to VLMQueue for intelligent routing.
    
    Lifecycle:
    1. Receives detection from VLMQueue (via "in" input)
    2. Sets busy flag
    3. Lazy-loads SmolVLM models if needed
    4. Runs VLM inference (5-30 seconds)
    5. Publishes enhanced alert to MQTT
    6. Clears busy flag
    
    Memory management:
    - Models loaded on first use
    - Can be unloaded via unload_model() for sequential loading
    
    Example:
        vlm_worker = SmolVLMVerifier(
            model_path="~/onnx/SmolVLM2-256M-Instruct",
            model_size="256M",
            device_id="mac-dev-01",
            mqtt_client=mqtt_client,
        )
        
        vlm_buffer = Buffer(size=1, policy=blocking_policy)
        vlm_queue.add_output("vlm", vlm_buffer)
        vlm_worker.add_input("in", vlm_buffer)
        runner.add_task(vlm_worker)
    """
    
    def __init__(
        self,
        model_path: str,
        model_size: str = "256M",
        device_id: str = None,
        max_new_tokens: int = 100,
        name: str = "smolvlm_verifier",
        result_callback: Optional[Callable[[str, str, bool, bool], None]] = None,
    ):
        """
        Initialize SmolVLM worker.

        Args:
            model_path: Path to ONNX model directory
            model_size: Model size (256M or 500M)
            device_id: Device identifier for alert metadata
            max_new_tokens: Maximum tokens to generate
            name: Task name
            result_callback: Called after each inference as
                callback(filter_id, track_id, confirmed, is_valid).
                Used to update tracker state in-memory without extra queues.
        """
        super().__init__(name=name)

        # Configuration
        self.model_path = Path(model_path).expanduser()
        self.model_size = model_size
        self.device_id = device_id
        self.max_new_tokens = max_new_tokens

        # SmolVLMOnnx wrapper — lazy-loaded on first process()
        self.model: Optional[SmolVLMOnnx] = None

        # In-memory callback to notify tracker of VLM result
        self.result_callback = result_callback

        # Busy state (thread-safe)
        self._busy = False
        self._busy_lock = threading.Lock()
        
        # Metrics
        self.stats = {
            'detections_processed': 0,
            'vlm_successes': 0,
            'vlm_failures': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0,
        }
    
    def is_busy(self) -> bool:
        """Check if worker is currently processing.
        
        Thread-safe busy state check for VLMQueue routing.
        
        Returns:
            True if processing, False if ready for new work
        """
        with self._busy_lock:
            return self._busy
    
    def _set_busy(self, busy: bool) -> None:
        """Set busy state (thread-safe)."""
        with self._busy_lock:
            self._busy = busy
    
    def is_ready(self) -> bool:
        """Override BaseTask.is_ready() to include busy check.
        
        Worker is ready if:
        1. Has input data (detection waiting)
        2. Not currently busy processing
        
        Returns:
            True if can process next detection
        """
        if self._busy:
            return False
        
        # Check if we have input
        has_input = any(buf.has_data() for buf in self.inputs.values()) if self.inputs else False
        return has_input
    
    def process(self) -> None:
        """Process next detection from input buffer."""
        # Get detection from VLMQueue
        input_buffer = list(self.inputs.values())[0]
        message = input_buffer.get()
        if not message:
            return
        
        # Extract confirmation event from message dict
        confirmations = message.get("confirmations", [])
        if not confirmations:
            logger.debug("No confirmations in message, skipping")
            return
        
        event = confirmations[0]  # Process first confirmation
        if not isinstance(event, dict):
            logger.warning(f"Received non-dict event: {type(event)}")
            return
        
        # Mark as busy (prevents VLMQueue from sending more work)
        self._set_busy(True)
        
        try:
            # Initialize models on first use
            self._initialize_models()
            
            # Run VLM verification
            start_time = time.time()
            vlm_result = self._run_vlm_verification(event)
            inference_time = time.time() - start_time
            
            # Update stats
            self.stats['detections_processed'] += 1
            self.stats['vlm_successes'] += 1
            self.stats['total_inference_time'] += inference_time
            self.stats['avg_inference_time'] = (
                self.stats['total_inference_time'] / self.stats['detections_processed']
            )
            
            # Only publish if confirmed OR invalid (not rejected)
            if vlm_result['confirmed']:
                self._publish_vlm_result(event, vlm_result, inference_time)
                logger.info(
                    f"[SmolVLM] CONFIRMED track {event.get('track_id')} "
                    f"filter={event.get('filter_id')} "
                    f"reply={vlm_result['text']!r} ({inference_time:.1f}s)"
                )
            elif not vlm_result['is_valid']:
                self._publish_vlm_result(event, vlm_result, inference_time)
                logger.warning(
                    f"[SmolVLM] INVALID RESPONSE track {event.get('track_id')} "
                    f"reply={vlm_result['text']!r} ({inference_time:.1f}s)"
                )
            else:
                logger.info(
                    f"[SmolVLM] REJECTED track {event.get('track_id')} "
                    f"filter={event.get('filter_id')} "
                    f"reply={vlm_result['text']!r} ({inference_time:.1f}s)"
                )

            # Notify tracker so it can transition the track state in-memory
            if self.result_callback:
                self.result_callback(
                    event.get('filter_id'),
                    event.get('track_id'),
                    vlm_result['confirmed'],
                    vlm_result['is_valid'],
                )
            
        except Exception as e:
            logger.error(f"VLM inference failed for track {event.get('track_id')}: {e}", exc_info=True)
            self.stats['vlm_failures'] += 1
            # Treat as invalid so the tracker can reset the track for a retry
            if self.result_callback:
                self.result_callback(event.get('filter_id'), event.get('track_id'), False, False)
            raise
        
        finally:
            # Mark as ready (VLMQueue can send next detection)
            self._set_busy(False)
    
    def _initialize_models(self) -> None:
        """Lazy-load SmolVLMOnnx wrapper on first use."""
        if self.model is not None:
            return

        logger.info(f"Loading SmolVLM {self.model_size} models from {self.model_path}...")
        start = time.time()
        try:
            self.model = SmolVLMOnnx(
                model_path=str(self.model_path),
                model_size=self.model_size,
                device="cuda",
            )
            logger.info(f"SmolVLM models loaded in {time.time() - start:.1f}s")
        except Exception as e:
            logger.error(f"Failed to load SmolVLM models: {e}", exc_info=True)
            raise
    
    def _run_vlm_verification(self, event: dict) -> dict:
        """Run SmolVLM inference on detection crop.

        Args:
            event: Detection event with crop_jpeg and vlm_reasoning

        Returns:
            Dict with keys: text, is_valid, confirmed
        """
        crop_jpeg = event.get("crop_jpeg")
        if not crop_jpeg:
            return {"text": "No image available", "is_valid": False, "confirmed": False}

        image = Image.open(io.BytesIO(crop_jpeg))

        vlm_reasoning = event.get("vlm_reasoning") or "person in the image"
        prompt = f"Is this: {vlm_reasoning}? Answer yes or no."
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        inputs = self.model.prepare_inputs(messages, [image])
        result = self.model.generate(inputs, max_new_tokens=self.max_new_tokens)

        result_lower = result.lower().strip()
        is_yes = result_lower.startswith("yes")
        is_no = result_lower.startswith("no")

        return {
            "text": result,
            "is_valid": is_yes or is_no,
            "confirmed": is_yes,
        }
    
    def _publish_vlm_result(self, event: dict, vlm_result: dict, inference_time: float) -> None:
        """Send VLM-verified alert to alert_publisher.
        
        Args:
            event: Detection event
            vlm_result: Structured VLM result with text, is_valid, confirmed
            inference_time: Time taken for inference
        """
        # Set VLM status based on result
        if vlm_result['is_valid']:
            if vlm_result['confirmed']:
                event["vlm_status"] = "confirmed"
                event["vlm_confirmed"] = True
            # Note: rejected (no) case won't reach here - filtered in process()
        else:
            # Invalid response - send as error alert
            event["vlm_status"] = "invalid_response"
            event["vlm_confirmed"] = False
            event["vlm_response"] = vlm_result['text']
        
        event["vlm_inference_time"] = inference_time
        
        # Send to alert_publisher via output buffer
        if self.outputs:
            message = {}
            message.setdefault("confirmations", []).append(event)
            output_buffer = list(self.outputs.values())[0]
            output_buffer.put(message)
        else:
            logger.warning("[SmolVLM] No output buffer configured - cannot publish VLM result")
    
    def warmup(self) -> None:
        """Eager-load ONNX models before the pipeline starts."""
        logger.info(f"[SmolVLM] Warming up {self.model_size} models...")
        self._initialize_models()
        logger.info("[SmolVLM] Warm-up complete — ready")

    def stop(self) -> None:
        """Unload models on pipeline shutdown."""
        if self.model is not None:
            self.unload_model()

    def unload_model(self) -> None:
        """Unload SmolVLM models to free GPU memory."""
        logger.info("Unloading SmolVLM models...")
        self.model = None
        import gc
        gc.collect()
        logger.info("SmolVLM models unloaded")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics.
        
        Returns:
            Dict with processing stats and busy state
        """
        return {
            **self.stats,
            'is_busy': self.is_busy(),
            'models_loaded': self.model is not None,
        }
