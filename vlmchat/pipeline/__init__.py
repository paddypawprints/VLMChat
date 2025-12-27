"""
VLMChat Pipeline Framework

A cursor-based execution engine for building data processing pipelines with DSL support.
"""

# Core framework
from .core import (
    BaseTask,
    Context,
    ContextDataType,
    Connector,
    LoopControlAction,
    register_task,
    get_task_registry,
    PipelineRunner,
    PipelineFactory,
    Environment,
)

# DSL
from .dsl import DSLParser, TaskHelpFormatter

# Cache system
from .cache import (
    CachedItem,
    CachedItemType,
    ItemCache,
    ImageContainer,
    EmbeddingContainer,
    TextContainer,
)

# Image processing
from .image import ImageFormat, ImageFormatConverter

# Connectors
from .connectors import (
    ForkConnector,
    LoopConnector,
    DetectionMergeConnector,
)

# Tasks
from .tasks import (
    DiagnosticTask,
    PassTask,
    StartTask,
    TimeoutTask,
    ClearTask,
    DetectorTask,
    YoloUltralyticsTask,
    YoloTensorRTTask,
)

# Trace
from .trace import BaseTrace, InMemoryTrace, LogTrace, NoOpTrace, print_trace_events

__all__ = [
    # Core
    "BaseTask",
    "Context",
    "ContextDataType",
    "Connector",
    "LoopControlAction",
    "register_task",
    "get_task_registry",
    "PipelineRunner",
    "PipelineFactory",
    "Environment",
    # DSL
    "DSLParser",
    "TaskHelpFormatter",
    # Cache
    "CachedItem",
    "CachedItemType",
    "ItemCache",
    "ImageContainer",
    "EmbeddingContainer",
    "TextContainer",
    # Image
    "ImageFormat",
    "ImageFormatConverter",
    # Connectors
    "ForkConnector",
    "LoopConnector",
    "DetectionMergeConnector",
    # Tasks
    "DiagnosticTask",
    "PassTask",
    "StartTask",
    "TimeoutTask",
    "ClearTask",
    "DetectorTask",
    "YoloUltralyticsTask",
    "YoloTensorRTTask",
    # Trace
    "BaseTrace",
    "InMemoryTrace",
    "LogTrace",
    "NoOpTrace",
    "print_trace_events",
]
