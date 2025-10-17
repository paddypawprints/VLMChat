#!/usr/bin/env python3
"""
Entry point for the SmolVLM chat application.

This module serves as the main entry point for the SmolVLM (Small Vision Language Model)
chat application. It loads configuration, sets up the Python path, and initializes
the chat application with configured parameters.
"""

import sys
import os
import argparse
from pathlib import Path
import logging
import traceback

# Ensure we're using the correct Python path for local imports
PROJECT_ROOT = Path(__file__).parent.resolve()
# Add both the repository root (parent of `src`) and the `src` directory to sys.path.
# This ensures imports like `src.prompt...` (which expect the repo root on sys.path)
# and imports like `main.console_io` (which expect `src` on sys.path) both work
# whether the module is executed as a script or imported as a package.
REPO_ROOT = PROJECT_ROOT.parent
for p in (str(REPO_ROOT), str(PROJECT_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from main.console_io import run_interactive_chat
from config import load_config, create_default_config_file, get_config


def resolve_config_candidate(config_arg: str | None) -> str | None:
    """Resolve which configuration file to use.

    Preference order when a config path is provided or omitted:
      1. If config_arg is provided, look for it relative to PROJECT_ROOT (src/),
         then REPO_ROOT, then as a literal path.
      2. If config_arg is None, attempt to find 'config.json' under
         PROJECT_ROOT, then REPO_ROOT. Return None if no file is found.
    Returns the selected path as a string or None.
    """
    from pathlib import Path
    if config_arg:
        candidates = [PROJECT_ROOT / config_arg, REPO_ROOT / config_arg, Path(config_arg)]
    else:
        candidates = [PROJECT_ROOT / "config.json", REPO_ROOT / "config.json"]

    for p in candidates:
        try:
            if p.exists():
                return str(p)
        except Exception:
            continue
    return None


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments
    """
    parser = argparse.ArgumentParser(
        description="SmolVLM Chat Application",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/main.py                           # Use default configuration
  python src/main.py --config config.json     # Use custom configuration file
  python src/main.py --create-config          # Create default config.json file
        """
    )

    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to configuration file (JSON or YAML)"
    )
    parser.add_argument(
        "--create-config",
        action="store_true",
        help="Create a default configuration file and exit"
    )
    parser.add_argument(
        "--config-output", "-o",
        type=str,
        default="config.json",
        help="Output path for created configuration file (default: config.json)"
    )
    parser.add_argument(
        "--onnx-info",
        action="store_true",
        help="Show ONNX model information and exit"
    )

    return parser.parse_args()


def main():
    """
    Main entry point for the SmolVLM chat application.

    Loads configuration, configures logging, initializes the chat application,
    and starts the interactive chat loop. Handles any initialization errors gracefully.

    Raises:
        Exception: Any unhandled exceptions during application startup are logged
                  and displayed to the user.
    """
    args = parse_arguments()

    # Handle config file creation
    if args.create_config:
        create_default_config_file(args.config_output)
        return

    # Handle ONNX info display
    if args.onnx_info:
        from utils.onnx_utils import get_onnx_model_info
        config = load_config(args.config) if args.config else load_config()
        info = get_onnx_model_info(config.model.model_path, config.model.onnx_base_path)

        print("ONNX Model Information:")
        print("=" * 50)
        print(f"  Model path: {info['model_path']}")
        print(f"  ONNX base path: {info['onnx_base_path']}")
        print(f"  ONNX model directory: {info['onnx_path']}")
        print(f"  Directory exists: {info['d/irectory_exists']}")
        print(f"  All ONNX files exist: {info['all_files_exist']}")

        if info['missing_files']:
            print(f"  Missing files: {', '.join(info['missing_files'])}")

        print(f"  Can use ONNX: {info['can_use_onnx']}")

        if not info['can_use_onnx']:
            print("\nTo enable ONNX inference:")
            print(f"  1. Create directory: {info['onnx_path']}")
            print("  2. Add required ONNX files:")
            for filename in info['required_files']:
                status = "✓" if filename.replace('.onnx', '') + '.onnx' not in info['missing_files'] else "✗"
                print(f"     {status} {filename}")

        return

    try:
        # Load configuration from file or environment variables
        if args.config:
            # Resolve config path preferentially relative to PROJECT_ROOT (src/),
            # then the repository root, then the literal provided path. This lets
            # users pass a simple 'config.json' while keeping a canonical
            # `src/config.json` as the authoritative source for development.
            from pathlib import Path
            candidates = [PROJECT_ROOT / args.config, REPO_ROOT / args.config, Path(args.config)]
            chosen = None
            for p in candidates:
                try:
                    if p.exists():
                        chosen = str(p)
                        break
                except Exception:
                    # ignore permission or other path-related errors and try next
                    continue
            if chosen is None:
                # Fall back to attempting to load the literal value; let
                # load_config raise a helpful error if it doesn't exist.
                chosen = args.config

            config = load_config(chosen)
            print(f"Loaded configuration from: {chosen}")
        else:
            # Try to load from environment variables or use defaults
            config = load_config()
            print("Using configuration from environment variables or defaults")

        # Configure application-wide logging using loaded configuration
        logging.basicConfig(
            level=getattr(logging, config.logging.level),
            format=config.logging.format
        )

        # Log environment information for debugging
        logging.info(f"Using Python: {sys.executable}")
        logging.info(f"Project root: {config.paths.project_root}")
        logging.info(f"Model path: {config.model.model_path}")
        logging.info(f"Max conversation pairs: {config.conversation.max_pairs}")
        logging.info(f"History format: {config.conversation.history_format}")
        logging.info(f"ONNX enabled: {config.model.use_onnx}")
        # Determine runtime platform: prefer explicit config value, else auto-detect
        try:
            if config.runtime_platform:
                logging.info(f"Runtime platform specified in config: {config.runtime_platform}")
            else:
                from utils.platform_detect import detect_platform
                detected = detect_platform()
                config.runtime_platform = detected.value
                logging.info(f"Detected runtime platform: {config.runtime_platform}")
        except Exception as e:
            logging.warning(f"Failed to determine runtime platform: {e}")

        # Start the interactive chat interface. console_io will create the
        # application instance and manage the I/O loop.
        run_interactive_chat()

    except Exception as e:
        # Log and display any startup errors
        logging.error(f"Application error: {e}")
        print(f"Failed to start application: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
