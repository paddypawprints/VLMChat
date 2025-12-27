#!/usr/bin/env python3
"""
Entry point for the SmolVLM chat application.

This module serves as the main entry point for the SmolVLM (Small Vision Language Model)
chat application. It loads configuration, sets up the Python path, and initializes
the chat application with configured parameters.
"""

import sys
import os

# Suppress warnings early - before any other imports
# This affects both current process and subprocesses via environment
if 'PYTHONWARNINGS' not in os.environ:
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::FutureWarning,ignore::DeprecationWarning,ignore::ResourceWarning'

import argparse
from pathlib import Path
import logging
import warnings

from utils.config import VLMChatConfig
from metrics.metrics_collector import Collector

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

from vlmchat.main.chat_console import run_interactive_chat
from vlmchat.utils.config import VLMChatConfig, create_default_config_file


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

    print("Starting SmolVLM Chat Application...")
    args = parse_arguments()

    # Handle config file creation

    if args.create_config:
        create_default_config_file(args.config_output)
        return

    path = None
    # if path is specified load from it, else try to resolve default locations
    if args.config:
        # Resolve config path preferentially relative to PROJECT_ROOT (src/),
        # then the repository root, then the literal provided path. This lets
        # users pass a simple 'config.json' while keeping a canonical
        # `src/config.json` as the authoritative source for development.
        from pathlib import Path
        p = Path(args.config)
        if p.is_absolute() and p.exists():
            path = str(p)
        else:   
            candidates = [PROJECT_ROOT / args.config, REPO_ROOT / args.config, Path(args.config)]
            for p in candidates:
                if p.exists():
                    path = str(p)
                    break
           
    if path is None:
        # Fall back to attempting to load the literal value; let
        # load_config raise a helpful error if it doesn't exist.
        config = VLMChatConfig()
        logging.info(f"Using default configuration")
    else:
        config = VLMChatConfig.load_from_file(path)
        logging.info(f"Loaded configuration from: {path}")
    # Configure application-wide logging using loaded configuration
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format
    )
    
    # Re-enable warnings and stderr if in DEBUG/INFO mode
    if config.logging.level in ('DEBUG', 'INFO'):
        os.environ['PYTHONWARNINGS'] = 'default'
        warnings.resetwarnings()
    else:
        # Redirect stderr to devnull to suppress all third-party warnings
        sys.stderr = open(os.devnull, 'w')

    # Log environment information for debugging
    logging.info(f"Using Python: {sys.executable}")
    logging.info(f"Project root: {config.paths.project_root}")
    logging.info(f"Model path: {config.model.model_path}")
    logging.info(f"Max conversation pairs: {config.conversation.max_pairs}")
    logging.info(f"History format: {config.conversation.history_format}")
    logging.info(f"ONNX enabled: {config.model.use_onnx}")
    # Determine runtime platform: prefer explicit config value, else auto-detect

    if config.platform:
        logging.info(f"Runtime platform specified in config: {config.platform}")
    else:
        from utils.platform_detect import detect_platform
        detected = detect_platform()
        config.runtime_platform = detected.value
        logging.info(f"Detected runtime platform: {config.platform}")

    collector = Collector(name = "VLMChat_metrics")
    # Start the interactive chat interface. console_io will create the
    # application instance and manage the I/O loop.
    run_interactive_chat(config, collector)

if __name__ == "__main__":
    main()
