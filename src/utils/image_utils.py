# utils/image_utils.py
"""
Utility functions for image processing and loading.

This module provides helper functions for loading images from various sources
including URLs and local file paths. All images are normalized to RGB format
for consistent processing by the vision model.
"""

import io
import logging
import requests
from PIL import Image
from typing import Optional

logger = logging.getLogger(__name__)

def load_image_from_url(url: str) -> Optional[Image.Image]:
    """
    Download and load an image from a URL.

    Downloads an image from the specified URL, validates it, and converts it to
    RGB format for consistent processing. Handles various image formats and
    provides detailed error logging for troubleshooting.

    Args:
        url: The URL of the image to download

    Returns:
        Optional[Image.Image]: The loaded PIL Image in RGB format, or None if loading failed

    Raises:
        requests.exceptions.RequestException: Network or HTTP errors during download
        Image.UnidentifiedImageError: If the downloaded data is not a valid image
    """
    try:
        # Download image with streaming to handle large files
        response = requests.get(url, stream=True)
        response.raise_for_status()

        # Load image from downloaded bytes
        image_bytes = io.BytesIO(response.content)
        pil_image = Image.open(image_bytes)

        # Ensure consistent RGB format for vision model
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        logger.info("Successfully loaded image from URL in RGB format")
        logger.debug(f"Image format: {pil_image.format}")
        logger.debug(f"Image mode: {pil_image.mode}")
        logger.debug(f"Image size: {pil_image.size}")

        return pil_image

    except requests.exceptions.RequestException as e:
        logger.error(f"Error downloading image from {url}: {e}")
        return None
    except Image.UnidentifiedImageError:
        logger.error(f"Could not identify image file from {url}")
        return None

def load_image_from_file(image_path: str) -> Optional[Image.Image]:
    """
    Load an image from a local file path.

    Opens and loads an image from the specified local file path, converting it
    to RGB format for consistent processing by the vision model.

    Args:
        image_path: Path to the local image file

    Returns:
        Optional[Image.Image]: The loaded PIL Image in RGB format, or None if loading failed

    Raises:
        Exception: File I/O errors or invalid image format errors
    """
    try:
        # Load image and ensure RGB format
        image = Image.open(image_path).convert('RGB')
        logger.info(f"Successfully loaded image from: {image_path}")
        logger.debug(f"Image size: {image.size}")
        return image
    except Exception as e:
        logger.error(f"Failed to load image from {image_path}: {e}")
        return None
