# utils/image_utils.py
"""Utility functions for image processing."""

import io
import logging
import requests
from PIL import Image
from typing import Optional

logger = logging.getLogger(__name__)

def load_image_from_url(url: str) -> Optional[Image.Image]:
    """
    Downloads an image from a URL and returns it as a PIL.Image object.
    The image is converted to 'RGB' mode.

    Args:
        url (str): The URL of the image.

    Returns:
        Optional[PIL.Image.Image]: The image object, or None if failed.
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        image_bytes = io.BytesIO(response.content)
        pil_image = Image.open(image_bytes)

        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')

        logger.info("Successfully created PIL Image in RGB format.")
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
    """Load an image from file path."""
    try:
        _image = Image.open(image_path).convert('RGB')
        logger.info(f"Successfully loaded image from: {image_path}")
        return _image
    except Exception as e:
        logger.error(f"Failed to load image from {image_path}: {e}")
        return None
