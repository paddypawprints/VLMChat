"""
Runtime platform detection utilities.

Provides a small heuristic to detect which platform and architecture the code is running on. Detection is intentionally conservative and allows
overrides via environment variables for testing.
"""
from __future__ import annotations

import os
from enum import Enum
import platform as _platform
from typing import Optional


class Platform(Enum):
    """Supported platforms."""
    RPI = "rpi"
    JETSON = "jetson"
    MAC = "mac"


def detect_platform(cfg : str = None) -> Optional[Platform]:
    """Detect runtime platform.

    Tries the following heuristics in order:
    1. Environment override: VLMCHAT_PLATFORM (rpi|jetson|mac|generic)
    2. Look for Jetson-specific files (/proc/device-tree/compatible) containing 'nvidia'
    3. Raspbian / Raspberry Pi hints in /proc/device-tree/model or platform.machine()
    4. Fallback to GENERIC mapped to RPI by default for now

    Returns:
        Platform: Detected platform enum
    """
    env = cfg if cfg else os.environ.get('VLMCHAT_PLATFORM')
    if env:
        env = env.strip().lower()
        if env in (Platform.RPI.value):
            return Platform.RPI
        if env in (Platform.JETSON.value):
            return Platform.JETSON
        if env in (Platform.MAC.value):
            return Platform.MAC

    match _platform.system().lower(): # type: ignore
        case 'darwin':
            return Platform.MAC
        case 'windows':
            raise NotImplementedError("Windows platform is not supported yet.")
        case 'linux':
            pass

    # Check for Jetson via device-tree compatible or /proc/cpuinfo markers
    try:
        # /proc/device-tree/compatible may contain 'nvidia' on Jetson
        if os.path.exists('/proc/device-tree/compatible'):
            with open('/proc/device-tree/compatible', 'rb') as f:
                content = f.read().lower()
                if b'nvidia' in content or b'jetson' in content:
                    return Platform.JETSON
    except Exception:
        pass

    # Additional Jetson-specific checks
    try:
        # Common NVIDIA Jetson release file
        if os.path.exists('/etc/nv_tegra_release'):
            return Platform.JETSON

        # /proc/cpuinfo may include 'tegra' on Jetson platforms
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r', encoding='utf-8', errors='ignore') as f:
                cpuinfo = f.read().lower()
                if 'tegra' in cpuinfo or 'nvidia' in cpuinfo:
                    return Platform.JETSON

        # Some Jetson systems expose tegra modules in sysfs
        if os.path.exists('/sys/module/tegra_fuse') or os.path.exists('/sys/module/tegra_si_drv'):
            return Platform.JETSON

        # Common Jetson utilities
        if os.path.exists('/usr/bin/tegrastats') or os.path.exists('/usr/bin/nvpmodel'):
            return Platform.JETSON
    except Exception:
        pass

    # Check model string for Raspberry Pi
    try:
        if os.path.exists('/proc/device-tree/model'):
            with open('/proc/device-tree/model', 'r', encoding='utf-8', errors='ignore') as f:
                model = f.read().lower()
                if 'raspberry' in model or 'imx' in model:
                    return Platform.RPI
    except Exception:
        pass

    # Additional Raspberry Pi checks
    try:
        # /etc/os-release may include raspbian or raspberry pi identifiers
        if os.path.exists('/etc/os-release'):
            with open('/etc/os-release', 'r', encoding='utf-8', errors='ignore') as f:
                osrel = f.read().lower()
                if 'raspbian' in osrel or 'raspios' in osrel or 'raspberry' in osrel:
                    return Platform.RPI

        # /proc/cpuinfo on older Pi's can include 'raspberry'
        if os.path.exists('/proc/cpuinfo'):
            with open('/proc/cpuinfo', 'r', encoding='utf-8', errors='ignore') as f:
                cpuinfo = f.read().lower()
                if 'raspberry' in cpuinfo:
                    return Platform.RPI
    except Exception:
        pass

    # Fallback to architecture-based heuristic
    arch = _platform.machine().lower()
    if arch.startswith('arm'):
        # ARM 32-bit is most likely Raspberry Pi
        return Platform.RPI
    if 'aarch64' in arch:
        # 64-bit ARM can be Jetson or newer Pi; if we reached here detection didn't find Jetson
        # prefer returning None so callers fall back to explicit defaults rather than misclassifying
        return Platform.RPI

    # Default fallback
    return None
