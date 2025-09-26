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

# Ensure we're using the correct Python path for local imports
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from main.chat_application import SmolVLMChatApplication
from config import load_config, create_default_config_file, get_config


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
        print(f"  Directory exists: {info['directory_exists']}")
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
            config = load_config(args.config)
            print(f"Loaded configuration from: {args.config}")
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

        # Initialize chat application (it will use the global configuration)
        app = SmolVLMChatApplication()

        # Start the interactive chat interface
        app.run_interactive_chat()

    except Exception as e:
        # Log and display any startup errors
        logging.error(f"Application error: {e}")
        print(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
