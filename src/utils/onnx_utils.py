"""
ONNX utilities for model path management and validation.

This module provides utilities for managing ONNX model files and directories,
including path creation and validation.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple

logger = logging.getLogger(__name__)


def create_onnx_directory(onnx_path: Path) -> bool:
    """
    Create ONNX model directory if it doesn't exist.

    Args:
        onnx_path: Path to ONNX model directory

    Returns:
        bool: True if directory exists or was created successfully
    """
    try:
        onnx_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"ONNX directory ready: {onnx_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to create ONNX directory {onnx_path}: {e}")
        return False


def validate_onnx_files(onnx_path: Path) -> Tuple[bool, List[str]]:
    """
    Validate that all required ONNX files exist.

    Args:
        onnx_path: Path to ONNX model directory

    Returns:
        tuple: (all_files_exist, missing_files)
    """
    required_files = [
        "vision_encoder.onnx",
        "embed_tokens.onnx",
        "decoder_model_merged.onnx"
    ]

    missing_files = []
    for filename in required_files:
        file_path = onnx_path / filename
        if not file_path.exists():
            missing_files.append(filename)

    return len(missing_files) == 0, missing_files


def get_onnx_model_info(model_path: str, onnx_base_path: str) -> Dict[str, any]:
    """
    Get information about ONNX model availability.

    Args:
        model_path: Original model path
        onnx_base_path: Base ONNX directory

    Returns:
        dict: Information about ONNX model availability
    """
    from config import get_config
    config = get_config()
    onnx_path = config.model.get_onnx_model_path()

    all_exist, missing = validate_onnx_files(onnx_path)

    return {
        "model_path": model_path,
        "onnx_base_path": onnx_base_path,
        "onnx_path": str(onnx_path),
        "directory_exists": onnx_path.exists(),
        "all_files_exist": all_exist,
        "missing_files": missing,
        "can_use_onnx": onnx_path.exists() and all_exist,
        "required_files": [
            "vision_encoder.onnx",
            "embed_tokens.onnx",
            "decoder_model_merged.onnx"
        ]
    }


def get_onnx_file_paths(onnx_path: Path) -> Dict[str, Path]:
    """
    Get paths to specific ONNX model files.

    Args:
        onnx_path: Path to ONNX model directory

    Returns:
        dict: Dictionary mapping file types to their paths
    """
    return {
        "vision_encoder": onnx_path / "vision_encoder.onnx",
        "embed_tokens": onnx_path / "embed_tokens.onnx",
        "decoder": onnx_path / "decoder_model_merged.onnx"
    }


def check_onnx_availability() -> bool:
    """
    Check if ONNX runtime is available.

    Returns:
        bool: True if ONNX runtime can be imported
    """
    try:
        import onnxruntime
        return True
    except ImportError:
        logger.warning("ONNX runtime not available")
        return False


def log_onnx_status(config) -> None:
    """
    Log comprehensive ONNX status information.

    Args:
        config: VLMChatConfig instance
    """
    info = get_onnx_model_info(config.model.model_path, config.model.onnx_base_path)
    onnx_available = check_onnx_availability()

    logger.info("ONNX Configuration Status:")
    logger.info(f"  ONNX Runtime Available: {onnx_available}")
    logger.info(f"  ONNX Enabled in Config: {config.model.use_onnx}")
    logger.info(f"  Model Path: {info['model_path']}")
    logger.info(f"  ONNX Base Path: {info['onnx_base_path']}")
    logger.info(f"  ONNX Model Directory: {info['onnx_path']}")
    logger.info(f"  Directory Exists: {info['directory_exists']}")
    logger.info(f"  All ONNX Files Present: {info['all_files_exist']}")

    if info['missing_files']:
        logger.warning(f"  Missing ONNX Files: {', '.join(info['missing_files'])}")

    can_use = onnx_available and config.model.use_onnx and info['can_use_onnx']
    logger.info(f"  Can Use ONNX: {can_use}")

    if not can_use:
        if not onnx_available:
            logger.info("  Reason: ONNX runtime not installed")
        elif not config.model.use_onnx:
            logger.info("  Reason: ONNX disabled in configuration")
        elif not info['can_use_onnx']:
            logger.info("  Reason: ONNX files not available, will use transformers fallback")


def setup_onnx_environment(config) -> bool:
    """
    Set up ONNX environment and validate configuration.

    Args:
        config: VLMChatConfig instance

    Returns:
        bool: True if ONNX can be used, False for fallback to transformers
    """
    # Check if ONNX is disabled in config
    if not config.model.use_onnx:
        logger.info("ONNX disabled in configuration, using transformers")
        return False

    # Check if ONNX runtime is available
    if not check_onnx_availability():
        logger.warning("ONNX runtime not available, falling back to transformers")
        return False

    # Get ONNX model path
    onnx_path = config.model.get_onnx_model_path()

    # Check if directory exists
    if not onnx_path.exists():
        logger.warning(f"ONNX model directory does not exist: {onnx_path}")
        logger.info("Create the directory and add ONNX files to enable ONNX inference")
        logger.info("Falling back to transformers inference")
        return False

    # Check if all required files exist
    all_exist, missing_files = validate_onnx_files(onnx_path)
    if not all_exist:
        logger.warning(f"Missing ONNX model files: {missing_files}")
        logger.info(f"Expected files in {onnx_path}:")
        for filename in ["vision_encoder.onnx", "embed_tokens.onnx", "decoder_model_merged.onnx"]:
            exists = (onnx_path / filename).exists()
            status = "✓" if exists else "✗"
            logger.info(f"  {status} {filename}")
        logger.info("Falling back to transformers inference")
        return False

    logger.info(f"ONNX models found and ready: {onnx_path}")
    return True