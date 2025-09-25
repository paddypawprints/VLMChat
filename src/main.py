#!/usr/bin/env python3
"""
Entry point for the SmolVLM chat application.

This module serves as the main entry point for the SmolVLM (Small Vision Language Model)
chat application. It configures logging, sets up the Python path, and initializes
the chat application with default parameters.
"""

import sys
from pathlib import Path
import logging

# Ensure we're using the correct Python path for local imports
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from main.chat_application import SmolVLMChatApplication


def main():
    """
    Main entry point for the SmolVLM chat application.

    Configures logging, initializes the chat application with default model settings,
    and starts the interactive chat loop. Handles any initialization errors gracefully.

    Raises:
        Exception: Any unhandled exceptions during application startup are logged
                  and displayed to the user.
    """
    try:
        # Configure application-wide logging with timestamp and level
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

        # Log environment information for debugging
        logging.info(f"Using Python: {sys.executable}")
        logging.info(f"Project root: {PROJECT_ROOT}")

        # Initialize chat application with HuggingFace SmolVLM model
        app = SmolVLMChatApplication(
            model_path="HuggingFaceTB/SmolVLM2-256M-Instruct"
        )

        # Start the interactive chat interface
        app.run_interactive_chat()

    except Exception as e:
        # Log and display any startup errors
        logging.error(f"Application error: {e}")
        print(f"Failed to start application: {e}")


if __name__ == "__main__":
    main()
