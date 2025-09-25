#!/usr/bin/env python3
# main.py
"""Entry point for the SmolVLM chat application."""

import sys
from pathlib import Path
import logging

# Ensure we're using the correct Python path
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

from main.chat_application import SmolVLMChatApplication

def main():
    """Main entry point."""
    try:
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        logging.info(f"Using Python: {sys.executable}")
        logging.info(f"Project root: {PROJECT_ROOT}")
        
        # Create and run the chat application
        app = SmolVLMChatApplication(
            model_path="HuggingFaceTB/SmolVLM2-256M-Instruct")
        
        app.run_interactive_chat()
        
    except Exception as e:
        logging.error(f"Application error: {e}")
        print(f"Failed to start application: {e}")

if __name__ == "__main__":
    main()
