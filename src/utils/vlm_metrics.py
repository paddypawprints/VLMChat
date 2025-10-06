"""
VLM-specific metrics collection for SmolVLM model performance tracking.

This module provides specialized metrics collection for Vision-Language Models,
particularly SmolVLM, with component-level instrumentation.
"""

from typing import Dict, Any, List, Optional
import time
from contextlib import contextmanager
from pathlib import Path

from .metrics_collector import MetricsCollector, create_metrics_collector


class VLMMetricsTracker:
    """
    Specialized metrics tracker for Vision-Language Models.

    Provides component-level metrics collection for VLM inference pipeline
    including vision encoding, text generation, and resource usage.
    """

    def __init__(self, model_instance):
        """
        Initialize VLM metrics tracker.

        Args:
            model_instance: The SmolVLM model instance to track
        """
        self.model = model_instance

        # Determine runtime type
        runtime = "onnx" if model_instance.use_onnx else "transformers"

        # Create metrics collector
        self.collector = create_metrics_collector(
            service_name="smol-vlm",
            model_path=str(model_instance.config.model_path),
            runtime=runtime
        )

        # Create metric instruments following OpenTelemetry conventions
        self._setup_metrics()

        # Generation tracking state
        self.current_generation = {}

    def _setup_metrics(self):
        """Setup all metric instruments."""

        # Input metrics (Counters - cumulative values)
        self.input_tokens_counter = self.collector.create_counter(
            "vlm.input.tokens.total",
            "Total number of input tokens processed",
            "tokens"
        )

        self.input_images_counter = self.collector.create_counter(
            "vlm.input.images.total",
            "Total number of input images processed",
            "images"
        )

        # Output metrics (Counters)
        self.output_tokens_counter = self.collector.create_counter(
            "vlm.output.tokens.total",
            "Total number of output tokens generated",
            "tokens"
        )

        # Component timing (Histograms - distribution of durations)
        self.vision_encoding_duration = self.collector.create_histogram(
            "vlm.vision.encoding.duration",
            "Duration of vision encoding step",
            "ms"
        )

        self.token_embedding_duration = self.collector.create_histogram(
            "vlm.text.embedding.duration",
            "Duration of token embedding lookup",
            "ms"
        )

        self.text_generation_duration = self.collector.create_histogram(
            "vlm.text.generation.duration",
            "Duration of text generation step",
            "ms"
        )

        self.total_inference_duration = self.collector.create_histogram(
            "vlm.inference.total.duration",
            "Total inference duration end-to-end",
            "ms"
        )

        # Performance metrics (Gauges - point-in-time values)
        self.tokens_per_second_gauge = self.collector.create_gauge(
            "vlm.performance.tokens_per_second",
            "Current tokens per second generation rate",
            "tokens/s"
        )

        self.memory_usage_gauge = self.collector.create_gauge(
            "vlm.resources.memory.usage",
            "Current memory usage",
            "bytes"
        )

        self.kv_cache_size_gauge = self.collector.create_gauge(
            "vlm.resources.kv_cache.size",
            "Key-value cache size",
            "bytes"
        )

        # Generation quality metrics (Histograms)
        self.generation_steps_histogram = self.collector.create_histogram(
            "vlm.generation.steps.count",
            "Number of generation steps per inference",
            "steps"
        )

        self.time_to_first_token = self.collector.create_histogram(
            "vlm.generation.time_to_first_token",
            "Time to generate first token",
            "ms"
        )

    def record_input_processing(self, input_tokens: int, input_images: int,
                               image_resolution: tuple = None):
        """
        Record input processing metrics.

        Args:
            input_tokens: Number of input tokens
            input_images: Number of input images
            image_resolution: Optional image resolution tuple (width, height)
        """
        attributes = {
            "component": "input_processing",
            "batch_size": 1
        }

        if image_resolution:
            attributes.update({
                "image_width": image_resolution[0],
                "image_height": image_resolution[1]
            })

        self.input_tokens_counter.add(input_tokens, attributes)
        self.input_images_counter.add(input_images, attributes)

    @contextmanager
    def track_vision_encoding(self, image_count: int = 1):
        """Context manager to track vision encoding performance."""
        attributes = {
            "component": "vision_encoder",
            "image_count": image_count
        }

        with self.vision_encoding_duration.time(attributes):
            yield

    @contextmanager
    def track_token_embedding(self, token_count: int = 1):
        """Context manager to track token embedding performance."""
        attributes = {
            "component": "token_embedder",
            "token_count": token_count
        }

        with self.token_embedding_duration.time(attributes):
            yield

    @contextmanager
    def track_text_generation_step(self, step_number: int):
        """Context manager to track individual generation step."""
        attributes = {
            "component": "text_decoder",
            "step": step_number
        }

        with self.text_generation_duration.time(attributes):
            yield

    @contextmanager
    def track_total_inference(self):
        """Context manager to track total inference time."""
        start_time = time.time()
        self.current_generation = {
            "start_time": start_time,
            "steps": 0,
            "first_token_time": None
        }

        try:
            with self.total_inference_duration.time({"component": "full_inference"}):
                yield self
        finally:
            # Record final metrics
            total_time = time.time() - start_time
            if self.current_generation["steps"] > 0:
                tokens_per_sec = self.current_generation["steps"] / total_time
                self.tokens_per_second_gauge.record(tokens_per_sec, {
                    "measurement_type": "final"
                })

                self.generation_steps_histogram.record(
                    self.current_generation["steps"],
                    {"inference_type": "complete"}
                )

            self.current_generation = {}

    def record_generation_step(self, token_id: int, step_number: int):
        """
        Record metrics for a single generation step.

        Args:
            token_id: ID of generated token
            step_number: Current generation step number
        """
        self.current_generation["steps"] = step_number

        # Record first token timing
        if step_number == 1 and "start_time" in self.current_generation:
            ttft_ms = (time.time() - self.current_generation["start_time"]) * 1000
            self.time_to_first_token.record(ttft_ms, {
                "token_id": token_id
            })

        # Update output token count
        self.output_tokens_counter.add(1, {
            "step": step_number,
            "token_id": token_id
        })

    def record_memory_usage(self, memory_bytes: int, memory_type: str = "process"):
        """Record memory usage metrics."""
        self.memory_usage_gauge.record(memory_bytes, {
            "memory_type": memory_type
        })

    def record_kv_cache_size(self, cache_size_bytes: int):
        """Record key-value cache size."""
        self.kv_cache_size_gauge.record(cache_size_bytes, {
            "cache_type": "key_value"
        })

    def export_metrics(self, output_dir: Optional[Path] = None) -> str:
        """
        Export collected metrics to JSON file.

        Args:
            output_dir: Optional output directory, defaults to ./metrics

        Returns:
            Path to the created metrics file
        """
        if output_dir is None:
            output_dir = Path("./metrics")

        return self.collector.export_to_json(output_dir)


class NullVLMMetricsTracker:
    """
    Null object implementation of VLMMetricsTracker.

    Does nothing when called, allowing metrics code to run without
    any performance overhead when metrics are disabled.
    """

    def record_input_processing(self, *args, **kwargs):
        pass

    def record_generation_step(self, *args, **kwargs):
        pass

    def record_memory_usage(self, *args, **kwargs):
        pass

    def record_kv_cache_size(self, *args, **kwargs):
        pass

    def export_metrics(self, *args, **kwargs):
        return None

    @contextmanager
    def track_vision_encoding(self, *args, **kwargs):
        yield

    @contextmanager
    def track_token_embedding(self, *args, **kwargs):
        yield

    @contextmanager
    def track_text_generation_step(self, *args, **kwargs):
        yield

    @contextmanager
    def track_total_inference(self):
        yield self


def create_vlm_metrics_tracker(model_instance, enabled: bool = True) -> VLMMetricsTracker:
    """
    Convenience function to create a VLM metrics tracker.

    Args:
        model_instance: SmolVLM model instance to track
        enabled: Whether to enable metrics collection

    Returns:
        VLMMetricsTracker instance if enabled, NullVLMMetricsTracker otherwise
    """
    if enabled:
        return VLMMetricsTracker(model_instance)
    else:
        return NullVLMMetricsTracker()