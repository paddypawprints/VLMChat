#!/usr/bin/env python3
"""
scripts/get_versions.py

Scan the current Python environment for the project's common dependencies and
print version pins suitable for pinning in `pyproject.toml` or a requirements
file. Useful when preparing fixed dependency versions for reproducible installs.

Usage:
    python scripts/get_versions.py
    python scripts/get_versions.py --format requirements
    python scripts/get_versions.py --format json --out pinned_versions.json

Formats:
  - plain: human readable table (default)
  - requirements: pip-style `package==version` lines
  - json: JSON mapping of package -> version

The script attempts to map importable module names to PyPI distribution names
for more reliable lookup.
"""
from __future__ import annotations

import argparse
import json
import importlib
import importlib.metadata
import sys
from typing import Dict, List, Tuple, Optional

# Mapping of a logical name -> (import_name, distribution_name)
# distribution_name can be None to try to infer from the import module
PACKAGES: List[Tuple[str, str, Optional[str]]] = [
    ("numpy", "numpy", "numpy"),
    ("Pillow", "PIL", "Pillow"),
    ("requests", "requests", "requests"),
    ("pydantic", "pydantic", "pydantic"),
    ("psutil", "psutil", "psutil"),
    ("PyYAML", "yaml", "PyYAML"),
    # Vision/ML extras
    ("transformers", "transformers", "transformers"),
    ("torch", "torch", "torch"),
    ("onnxruntime", "onnxruntime", "onnxruntime"),
    # Platform-specific / optional
    ("picamera2", "picamera2", "picamera2"),
    ("jetson-stats", "jetson_stats", "jetson-stats"),
    # Dev/test
    ("pytest", "pytest", "pytest"),
    ("pytest-xdist", "xdist", "pytest-xdist"),
    ("pytest-timeout", "pytest_timeout", "pytest-timeout"),
    ("pytest-benchmark", "pytest_benchmark", "pytest-benchmark"),
]


def get_version_by_distribution(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def get_version_by_module(import_name: str) -> Optional[str]:
    try:
        module = importlib.import_module(import_name)
    except Exception:
        return None

    # Common version attributes
    for attr in ("__version__", "version", "VERSION"):
        v = getattr(module, attr, None)
        if isinstance(v, str):
            return v
    # Fallback: try distribution name
    return None


def detect_version(import_name: str, dist_name: Optional[str]) -> Optional[str]:
    # First try distribution metadata if we have a distribution name
    if dist_name:
        v = get_version_by_distribution(dist_name)
        if v:
            return v
    # Fallback to importing module and checking common attributes
    v = get_version_by_module(import_name)
    if v:
        return v
    # Last resort: try distribution lookup using import_name as dist
    try:
        v = get_version_by_distribution(import_name)
        if v:
            return v
    except Exception:
        pass
    return None


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dump versions of project deps")
    parser.add_argument("--format", choices=("plain", "requirements", "json"), default="plain")
    parser.add_argument("--out", help="Write JSON output to file (only used with --format json)")
    args = parser.parse_args(argv)

    results: Dict[str, Optional[str]] = {}
    for nice_name, import_name, dist_name in PACKAGES:
        ver = detect_version(import_name, dist_name)
        results[nice_name] = ver

    if args.format == "plain":
        print("Detected package versions (or 'not installed'):\n")
        longest = max(len(k) for k in results)
        for pkg, ver in results.items():
            print(f"{pkg.ljust(longest)} : {ver or 'not installed'}")
        print("\nTo generate pip-style pins, run with --format requirements")

    elif args.format == "requirements":
        lines: List[str] = []
        for pkg, ver in results.items():
            if ver:
                # convert nice_name to pip/pyproject package name where sensible
                # Use the distribution name where known (simple mapping)
                pip_name = pkg
                # Normalize a few common names
                if pkg == 'Pillow':
                    pip_name = 'Pillow'
                elif pkg == 'PyYAML':
                    pip_name = 'PyYAML'
                elif pkg == 'picamera2':
                    pip_name = 'picamera2'
                elif pkg == 'jetson-stats':
                    pip_name = 'jetson-stats'
                lines.append(f"{pip_name}=={ver}")
        print("# pip-style requirement pins")
        print("\n".join(lines))

    elif args.format == "json":
        out = {k: v for k, v in results.items()}
        json_text = json.dumps(out, indent=2)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(json_text)
            print(f"Wrote JSON to {args.out}")
        else:
            print(json_text)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
