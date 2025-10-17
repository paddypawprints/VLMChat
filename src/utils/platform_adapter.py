"""
Platform adapter layer for optional GPU metrics collection.

This module provides a small adapter abstraction allowing platform-specific
GPU metrics collection (Jetson, Raspberry Pi, generic no-op). The discovery
is performed by `detect_platform()` in `platform_detect.py` and callers can
obtain an adapter via `get_adapter()`.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from .platform_detect import detect_platform
from .camera_base import Platform

logger = logging.getLogger(__name__)


class PlatformAdapter(ABC):
    """Abstract base for platform adapters that may provide GPU metrics."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def get_gpu_metrics(self) -> Optional[Dict[str, float]]:
        """Return GPU metrics as a dict or None if not available.

        Example return value: {"util": 23.4, "mem_used": 512.0, "mem_total": 4096.0}
        """
        pass


class NoopAdapter(PlatformAdapter):
    @property
    def name(self) -> str:
        return "noop"

    def get_gpu_metrics(self) -> Optional[Dict[str, float]]:
        return None


class JetsonAdapter(PlatformAdapter):
    """Best-effort Jetson adapter using pynvml if available.

    Falls back to no-op if pynvml is not present or fails to initialize.
    """

    def __init__(self) -> None:
        # lazily import pynvml
        try:
            import pynvml
            self._pynvml = pynvml
            try:
                self._pynvml.nvmlInit()
                self._initialized = True
            except Exception:
                logger.exception("pynvml present but failed to initialize")
                self._initialized = False
        except Exception:
            self._pynvml = None
            self._initialized = False

    @property
    def name(self) -> str:
        return "jetson"

    def get_gpu_metrics(self) -> Optional[Dict[str, float]]:
        if not self._initialized or not self._pynvml:
            return None
        try:
            # support single-GPU devices (Jetson typically has one)
            handle = self._pynvml.nvmlDeviceGetHandleByIndex(0)
            util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
            mem = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
            return {
                "util": float(util.gpu),
                "mem_used": float(mem.used) / (1024.0 * 1024.0),
                "mem_total": float(mem.total) / (1024.0 * 1024.0),
            }
        except Exception:
            logger.exception("Failed to read pynvml GPU metrics")
            return None


def get_adapter(platform: Optional[Platform] = None) -> PlatformAdapter:
    """Return the best adapter for the current platform (or provided one)."""
    try:
        plat = platform or detect_platform()
    except Exception:
        plat = None

    if plat == Platform.JETSON:
        return JetsonAdapter()

    return NoopAdapter()
