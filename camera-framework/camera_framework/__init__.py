"""Camera Framework - Lightweight pipeline framework for edge device AI vision."""

from .task import BaseTask
from .runner import Runner
from .buffer import (
    Buffer,
    blocking_policy,
    drop_oldest_policy,
    drop_newest_policy,
    decimate_policy,
)
from .visitor import (
    PipelineTraverser,
    MermaidVisitor,
)
from .memory_monitor import (
    MemoryMonitor,
    memory_monitor,
    track_context,
    track_image,
)
from .metrics import (
    Collector,
    Instrument,
    AvgDurationInstrument,
    RecentSamplesInstrument,
    CounterInstrument,
    RateInstrument,
    MemoryInstrument,
    DurationTimer,
    null_collector,
)
from .detection import Detection, CocoCategory, ImageFormat

__version__ = "0.1.0"

__all__ = [
    "Context",
    "BaseTask",
    "Runner",
    "Buffer",
    "blocking_policy",
    "drop_oldest_policy",
    "drop_newest_policy",
    "decimate_policy",
    "Collector",
    "Instrument",
    "AvgDurationInstrument",
    "RecentSamplesInstrument",
    "CounterInstrument",
    "RateInstrument",
    "DurationTimer",
    "null_collector",
    "Detection",
    "CocoCategory",
    "ImageFormat",
]

# Bridges and cameras available as submodules
from . import bridges
from . import cameras
