#!/usr/bin/env python3
"""
Run the CLIP detection pipeline.

Loads clip_detection.dsl and executes it with proper setup.
"""

import sys
from pathlib import Path
import logging

# Set up Python paths correctly
PROJECT_ROOT = Path(__file__).parent.resolve()
SRC_DIR = PROJECT_ROOT / "src"
for p in [str(PROJECT_ROOT), str(SRC_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from vlmchat.utils.config import VLMChatConfig
from vlmchat.pipeline.dsl_parser import DSLParser, create_task_registry
from vlmchat.pipeline.pipeline_runner import PipelineRunner
from vlmchat.pipeline.task_base import Context
from vlmchat.metrics.metrics_collector import Collector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run the CLIP detection pipeline."""
    print("\n" + "="*70)
    print("CLIP Detection Pipeline")
    print("="*70)
    print("\nThis pipeline will:")
    print("  1. Prompt you for text queries (type exit or Ctrl+D to finish)")
    print("  2. Capture an image from your camera")
    print("  3. Detect objects with YOLO")
    print("  4. Filter relevant categories")
    print("  5. Encode text and images with CLIP")
    print("  6. Compare and rank similarities")
    print("\nMake sure:")
    print("  - Camera is connected")
    print("  - YOLO model (yolov8n.pt) is present")
    print("  - CLIP model is configured")
    print("="*70 + "\n")
    
    try:
        # Load configuration
        logger.info("Loading configuration...")
        config = VLMChatConfig()
        collector = Collector()
        
        # Load task registry
        logger.info("Loading task registry...")
        registry = create_task_registry()
        logger.info(f"Loaded {len(registry)} tasks")
        
        # Read DSL file
        dsl_path = PROJECT_ROOT / "pipelines" / "clip_detection.dsl"
        logger.info(f"Reading DSL: {dsl_path}")
        with open(dsl_path, 'r') as f:
            dsl_text = f.read()
        
        # Parse and build pipeline
        logger.info("Building pipeline...")
        parser = DSLParser(task_registry=registry)
        pipeline = parser.parse(dsl_text)
        logger.info(f"Pipeline built: {type(pipeline).__name__}")
        
        # Create runner with the pipeline
        runner = PipelineRunner(pipeline)
        
        # Execute pipeline (runner handles context creation and I/O)
        logger.info("Starting pipeline execution...\n")
        runner.run()
        
        logger.info("\nPipeline completed successfully")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
