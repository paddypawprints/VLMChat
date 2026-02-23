"""SmolVLM worker for async VLM verification."""

import base64
import logging
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Any, Dict
from PIL import Image
import io
import numpy as np

from camera_framework import BaseTask


logger = logging.getLogger(__name__)


class SmolVLMWorker(BaseTask):
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
        vlm_worker = SmolVLMWorker(
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
        name: str = "smolvlm_worker",
    ):
        """
        Initialize SmolVLM worker.
        
        Args:
            model_path: Path to ONNX model directory
            model_size: Model size (256M or 500M)
            device_id: Device identifier for alert metadata
            max_new_tokens: Maximum tokens to generate
            name: Task name
        """
        super().__init__(name=name)
        
        # Configuration
        self.model_path = Path(model_path).expanduser()
        self.model_size = model_size
        self.device_id = device_id
        self.max_new_tokens = max_new_tokens
        
        # Model backends (lazy-loaded)
        self.vision = None
        self.embed = None
        self.decoder = None
        self.processor = None
        self.tokenizer = None
        self.config = None
        self.image_token_id = None
        self.eos_token_id = []
        
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
            if vlm_result['confirmed'] or not vlm_result['is_valid']:
                self._publish_vlm_result(event, vlm_result, inference_time)
                status = "confirmed" if vlm_result['confirmed'] else "invalid"
                logger.info(
                    f"VLM {status} for track {event.get('track_id')}: {vlm_result['text']} "
                    f"({inference_time:.1f}s)"
                )
            else:
                logger.info(
                    f"VLM rejected track {event.get('track_id')}: {vlm_result['text']} "
                    f"({inference_time:.1f}s) - alert not sent"
                )
            
        except Exception as e:
            logger.error(f"VLM inference failed for track {event.get('track_id')}: {e}", exc_info=True)
            self.stats['vlm_failures'] += 1
            raise
        
        finally:
            # Mark as ready (VLMQueue can send next detection)
            self._set_busy(False)
    
    def _initialize_models(self) -> None:
        """Lazy-load SmolVLM models on first use.
        
        Memory management:
        - Vision encoder: GPU
        - Embed tokens: GPU
        - Decoder: CPU for 256M (GPU OOM on 8GB Jetson), GPU for 500M+
        """
        if self.vision is not None:
            return  # Already initialized
        
        logger.info(f"Loading SmolVLM {self.model_size} models...")
        start_time = time.time()
        
        try:
            # Import transformers for processor
            from transformers import AutoProcessor, AutoConfig
            
            # Determine HuggingFace model name
            hf_model_name = f"HuggingFaceTB/SmolVLM2-{self.model_size}-Instruct"
            
            # Load processor and config
            self.config = AutoConfig.from_pretrained(hf_model_name)
            self.processor = AutoProcessor.from_pretrained(hf_model_name)
            self.tokenizer = self.processor.tokenizer
            
            # Extract model parameters
            text_config = self.config.text_config
            num_hidden_layers = text_config.num_hidden_layers
            num_key_value_heads = text_config.num_key_value_heads
            head_dim = text_config.head_dim
            self.image_token_id = self.config.image_token_id
            
            # Handle EOS token IDs
            cfg_eos = text_config.eos_token_id
            if cfg_eos is not None:
                if isinstance(cfg_eos, (list, tuple)):
                    self.eos_token_id = list(cfg_eos)
                else:
                    self.eos_token_id = [int(cfg_eos)]
            else:
                self.eos_token_id = []
            
            if self.tokenizer.eos_token_id not in self.eos_token_id:
                self.eos_token_id.append(self.tokenizer.eos_token_id)
            
            # Load ONNX backends (local to macos-device, no vlmchat dependency)
            from .models.smolvlm_vision_onnx import SmolVLMVisionOnnx
            from .models.smolvlm_embed_onnx import SmolVLMEmbedOnnx
            from .models.smolvlm_decoder_onnx import SmolVLMDecoderOnnx
            
            self.vision = SmolVLMVisionOnnx(
                str(self.model_path / "vision_encoder.onnx"),
                device="cuda"
            )
            
            self.embed = SmolVLMEmbedOnnx(
                str(self.model_path / "embed_tokens.onnx"),
                device="cuda"
            )
            
            # Decoder device based on model size (256M → CPU, 500M → GPU)
            decoder_device = "cpu" if self.model_size == "256M" else "cuda"
            self.decoder = SmolVLMDecoderOnnx(
                str(self.model_path / "decoder_model_merged.onnx"),
                num_hidden_layers=num_hidden_layers,
                num_key_value_heads=num_key_value_heads,
                head_dim=head_dim,
                device=decoder_device,
                model_size=self.model_size,
            )
            
            load_time = time.time() - start_time
            logger.info(f"SmolVLM models loaded in {load_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Failed to load SmolVLM models: {e}", exc_info=True)
            raise
    
    def _run_vlm_verification(self, event: dict) -> str:
        """Run SmolVLM inference on detection crop.
        
        Args:
            event: Detection event with crop_jpeg and vlm_reasoning
            
        Returns:
            VLM generated text
        """
        # Decode crop image
        crop_jpeg = event.get("crop_jpeg")
        if not crop_jpeg:
            return "No image available for verification"
        
        image = Image.open(io.BytesIO(crop_jpeg))
        
        # Build prompt from vlm_reasoning
        vlm_reasoning = event.get("vlm_reasoning", "")
        if not vlm_reasoning:
            vlm_reasoning = "person in the image"
        
        prompt = f"Is this: {vlm_reasoning}? Answer yes or no."
        
        # Prepare inputs (similar to smol_batch.py)
        messages = [{"role": "user", "content": prompt}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        
        # Process image and text
        pixel_inputs = self.processor(images=[image], return_tensors="np")
        pixel_values = pixel_inputs['pixel_values']
        pixel_attention_mask = pixel_inputs['pixel_attention_mask']
        
        # Encode vision features
        image_features = self.vision(pixel_values, pixel_attention_mask)
        
        # Tokenize text
        input_ids = self.tokenizer.encode(text, return_tensors="np")[0]
        
        # Build input embeddings (merge image features and text)
        embeddings_list = []
        for token_id in input_ids:
            if token_id == self.image_token_id:
                embeddings_list.append(image_features[0])
            else:
                token_embed = self.embed(np.array([[token_id]], dtype=np.int64))
                embeddings_list.append(token_embed[0, 0])
        
        input_embeds = np.stack(embeddings_list, axis=0)[np.newaxis, ...]
        
        # Generate tokens
        generated_ids = []
        for _ in range(self.max_new_tokens):
            logits, past_key_values = self.decoder(input_embeds)
            next_token_id = int(np.argmax(logits[0, -1]))
            
            if next_token_id in self.eos_token_id:
                break
            
            generated_ids.append(next_token_id)
            
            # Embed next token
            next_embed = self.embed(np.array([[next_token_id]], dtype=np.int64))
            input_embeds = next_embed
        
        # Decode generated text
        result = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse yes/no response
        result_lower = result.lower().strip()
        is_yes = result_lower in ["yes", "yes."]
        is_no = result_lower in ["no", "no."]
        is_valid = is_yes or is_no
        
        return {
            "text": result,
            "is_valid": is_valid,
            "confirmed": is_yes
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
            logger.info(f"Sent VLM-verified alert for track {event.get('track_id')} to alert_publisher")
        else:
            logger.warning("No output buffer configured - cannot publish VLM result")
    
    def _publish_vlm_error(self, event: dict, error: str) -> None:
        """Send alert with VLM error flag to alert_publisher.
        
        Args:
            event: Detection event
            error: Error message
        """
        # Mark VLM failure in event
        event["vlm_verified"] = False
        event["vlm_error"] = error
        
        # Send to alert_publisher via output buffer
        if self.outputs:
            message = {}
            message.setdefault("confirmations", []).append(event)
            output_buffer = list(self.outputs.values())[0]
            output_buffer.put(message)
            logger.info(f"Sent VLM error alert for track {event.get('track_id')} to alert_publisher")
        else:
            logger.warning("No output buffer configured - cannot publish VLM error")
    
    def _format_bbox(self, bbox) -> Optional[dict]:
        """Format bounding box for JSON."""
        if not bbox:
            return None
        
        x1, y1, x2, y2 = bbox
        return {
            "x": float(x1),
            "y": float(y1),
            "width": float(x2 - x1),
            "height": float(y2 - y1),
        }
    
    def unload_model(self) -> None:
        """Unload SmolVLM models to free GPU memory.
        
        For sequential loading pattern where YOLO/CLIP are unloaded
        before SmolVLM is loaded.
        """
        logger.info("Unloading SmolVLM models...")
        
        self.vision = None
        self.embed = None
        self.decoder = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        logger.info("SmolVLM models unloaded")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get worker statistics.
        
        Returns:
            Dict with processing stats and busy state
        """
        return {
            **self.stats,
            'is_busy': self.is_busy(),
            'models_loaded': self.vision is not None,
        }
