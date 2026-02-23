"""Color extraction and matching utilities for detection filtering."""

import logging
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# HSV color ranges (Hue in degrees 0-360, Saturation/Value 0-100)
HSV_COLOR_RANGES = {
    'red': [(0, 15), (345, 360)],  # Red wraps around
    'orange': [(15, 45)],
    'yellow': [(45, 75)],
    'green': [(75, 155)],
    'cyan': [(155, 200)],
    'blue': [(200, 260)],
    'purple': [(260, 300)],
    'pink': [(300, 345)],
    'white': [],  # Special case: high value, low saturation
    'black': [],  # Special case: low value
    'gray': [],   # Special case: low saturation, mid value
}

# Thresholds for color matching
MIN_SATURATION = 50  # 0-100 scale
MIN_VALUE = 50       # 0-100 scale
MIN_PIXEL_PERCENTAGE = 5  # Minimum % of pixels to consider dominant (lowered for clothing patterns)


def extract_dominant_color(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    k: int = 3,
    min_pixel_pct: float = MIN_PIXEL_PERCENTAGE
) -> Optional[str]:
    """
    Extract dominant color from image region by counting pixels matching each color.
    
    Returns the color with the highest pixel percentage if it exceeds the threshold.
    
    Args:
        image: PIL Image
    bbox: Bounding box (x1, y1, x2, y2)
    k: Unused (kept for API compatibility)
    min_pixel_pct: Minimum pixel percentage threshold
        Returns:
            Color name (red, blue, etc.) or None if no clear color
    """
    # Crop region
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return None
    
    crop = image.crop((x1, y1, x2, y2))
    
    # Convert to numpy array and reshape
    img_array = np.array(crop)
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return None
    
    pixels = img_array.reshape(-1, 3)
    
    # Convert RGB to HSV
    hsv = rgb_to_hsv_array(pixels)
    total_pixels = len(hsv)
    
    # Vectorized color classification
    color_counts = count_color_pixels(hsv)
    
    # Find color with highest percentage
    max_color = None
    max_pct = 0.0
    
    for color, count in color_counts.items():
        pct = (count / total_pixels) * 100
        if pct > max_pct:
            max_pct = pct
            max_color = color
    
    # Log result
    logger.debug(f"Color extraction: {max_color} @ {max_pct:.1f}% (threshold: {min_pixel_pct}%)")
    
    # Return color if it meets threshold
    if max_color and max_pct >= min_pixel_pct:
        return max_color
    
    return None


def check_target_rgb_simple(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_rgb: Tuple[int, int, int],
    brightness_tolerance: int = 100,
    min_brightness: int = 120,
    max_color_diff: int = 40,
    min_confidence: float = 20.0,
    use_ellipse: bool = True,
    ellipse_margin: float = 0.05,
    use_lab: bool = True,
) -> Tuple[bool, float]:
    """Check if target RGB exists using LAB color space (robust to lighting variations).
    
    For achromatic colors (white/gray/black), checks:
        - Brightness range (all RGB values within tolerance of target)
    - Color consistency (RGB channels similar = achromatic)
    
    For white specifically, we check pixels are:
        - Bright enough (avg RGB >= min_brightness, default 150)
    - Achromatic (max-min RGB <= max_color_diff, default 40)
    - Similar brightness to target (within brightness_tolerance, default 80)
    
    This handles real-world white clothing with shadows/wrinkles much better than HSV.
    
    Args:
        image: PIL Image
    bbox: Bounding box (x1, y1, x2, y2)
    target_rgb: RGB tuple from Groq (r, g, b) each 0-255
    brightness_tolerance: How much darker/brighter pixels can be (default: 80)
    min_brightness: Minimum brightness for white (default: 150)
    max_color_diff: Max difference between RGB channels for achromatic (default: 40)
    min_confidence: Minimum percentage of matching pixels (default: 30%)
    use_ellipse: If True, check inscribed ellipse only
    ellipse_margin: Margin as fraction (default: 0.05 = 5%)
        Returns:
        (matches, confidence) - True if color present >= threshold
    """
    # Crop region
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return False, 0.0
    crop = image.crop((x1, y1, x2, y2))
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # Convert to numpy array
    img_array = np.array(crop)
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return False, 0.0
    
    # Create ellipse mask if requested
    if use_ellipse:
        center_x = crop_width / 2
        center_y = crop_height / 2
        a = (crop_width / 2) * (1 - ellipse_margin)
        b = (crop_height / 2) * (1 - ellipse_margin)
        y_coords, x_coords = np.ogrid[:crop_height, :crop_width]
        mask = ((x_coords - center_x) / a) ** 2 + ((y_coords - center_y) / b) ** 2 <= 1
        masked_pixels = img_array[mask]
        if len(masked_pixels) == 0:
            return False, 0.0
    else:
        masked_pixels = img_array.reshape(-1, 3)
    
    # Get RGB channels
    r, g, b = masked_pixels[:, 0], masked_pixels[:, 1], masked_pixels[:, 2]
    target_r, target_g, target_b = target_rgb
    
    # DEBUG: Log sample pixel values from the region
    sample_size = min(10, len(masked_pixels))
    sample_pixels = masked_pixels[:sample_size]
    logger.debug(f"DEBUG Color Check - Target RGB: {target_rgb}")
    logger.debug(f"DEBUG Color Check - Sample pixels from region ({sample_size} of {len(masked_pixels)}):")
    for i, pixel in enumerate(sample_pixels):
        logger.debug(f"  Pixel {i}: R={pixel[0]}, G={pixel[1]}, B={pixel[2]}")
    
    # Compute stats for the region
    mean_r, mean_g, mean_b = np.mean(r), np.mean(g), np.mean(b)
    logger.debug(f"DEBUG Color Check - Region mean RGB: R={mean_r:.1f}, G={mean_g:.1f}, B={mean_b:.1f}")
    
    # Detect if target is achromatic (white, gray, black)
    target_range = max(target_r, target_g, target_b) - min(target_r, target_g, target_b)
    is_achromatic = target_range < 30
    logger.debug(f"DEBUG Color Check - Target range: {target_range}, is_achromatic: {is_achromatic}")
    
    if is_achromatic:
        # For white/gray/black: just check if pixels are achromatic and in brightness range
        # Don't overthink it - white clothing in shadows is still "white"
        avg_brightness = (r.astype(np.float32) + g.astype(np.float32) + b.astype(np.float32)) / 3
        
        # Pixels are achromatic (RGB channels similar - this is the key for white)
        pixel_range = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        achromatic_match = pixel_range <= max_color_diff
        
        # DEBUG: Log achromatic matching stats
        logger.debug(f"DEBUG Color Check - Achromatic check (max_color_diff={max_color_diff}):")
        logger.debug(f"  Mean pixel range (R-G-B diff): {np.mean(pixel_range):.1f}")
        logger.debug(f"  Pixels with range <= {max_color_diff}: {np.sum(achromatic_match)} / {len(masked_pixels)} ({np.sum(achromatic_match)/len(masked_pixels)*100:.1f}%)")
        
        # For white: bright enough (relaxed - shadows make white darker)
        # For gray/black: check brightness range
        target_brightness = (target_r + target_g + target_b) / 3
        if target_brightness > 200:  # Target is white
            bright_enough = avg_brightness >= 100  # Very permissive - white in shadow
            matches = achromatic_match & bright_enough
            logger.debug(f"DEBUG Color Check - White target (brightness={target_brightness:.0f}):")
            logger.debug(f"  Mean brightness in region: {np.mean(avg_brightness):.1f}")
            logger.debug(f"  Pixels bright enough (>= 100): {np.sum(bright_enough)} / {len(masked_pixels)} ({np.sum(bright_enough)/len(masked_pixels)*100:.1f}%)")
            logger.debug(f"  Final matches (achromatic & bright): {np.sum(matches)} / {len(masked_pixels)} ({np.sum(matches)/len(masked_pixels)*100:.1f}%)")
        else:  # Gray or black
            brightness_match = np.abs(avg_brightness - target_brightness) <= brightness_tolerance
            matches = brightness_match & achromatic_match
            logger.debug(f"DEBUG Color Check - Gray/Black target (brightness={target_brightness:.0f}):")
            logger.debug(f"  Brightness matches: {np.sum(brightness_match)} / {len(masked_pixels)}")
        
        match_count = np.sum(matches)
        confidence = (match_count / len(masked_pixels)) * 100
        result = confidence >= min_confidence
        
        logger.debug(f"Simple RGB check: target={target_rgb} (achromatic, brightness={target_brightness:.0f}), "
                   f"matched={match_count}/{len(masked_pixels)} ({confidence:.1f}%), threshold={min_confidence}%, result={result}")
    else:
        # For chromatic colors: simple Euclidean distance
        distance = np.sqrt((r.astype(np.float32) - target_r)**2 + (g.astype(np.float32) - target_g)**2 + (b.astype(np.float32) - target_b)**2)
        matches = distance <= brightness_tolerance
        match_count = np.sum(matches)
        confidence = (match_count / len(masked_pixels)) * 100
        result = confidence >= min_confidence
        
        logger.debug(f"Simple RGB check: target={target_rgb} (chromatic), "
                   f"matched={match_count}/{len(masked_pixels)} ({confidence:.1f}%), threshold={min_confidence}%, result={result}")
    
    return result, confidence


def check_target_rgb_hsv(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_rgb: Tuple[int, int, int],
    hue_tolerance: float = 20,
    sat_tolerance: float = 50,
    val_tolerance: float = 50,
    min_confidence: float = 30.0,
    use_ellipse: bool = True,
    ellipse_margin: float = 0.05
) -> Tuple[bool, float]:
    """Check if a target RGB color (as HSV) is present in region.
    
    More precise than named colors - matches Groq's exact RGB with HSV tolerance.
    
    Args:
        image: PIL Image
        bbox: Bounding box (x1, y1, x2, y2)
        target_rgb: RGB tuple from Groq (r, g, b) each 0-255
        hue_tolerance: Hue tolerance in degrees (default: 15°)
        sat_tolerance: Saturation tolerance 0-100 (default: 25)
        val_tolerance: Value tolerance 0-100 (default: 25)
        min_confidence: Minimum percentage of matching pixels (default: 50%)
        use_ellipse: If True, check inscribed ellipse only
        ellipse_margin: Margin as fraction (default: 0.05 = 5%)
    
    Returns:
        (matches, confidence) - True if color present >= threshold
    """
    # Crop region
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return False, 0.0
    
    crop = image.crop((x1, y1, x2, y2))
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # Convert to numpy array
    img_array = np.array(crop)
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return False, 0.0
    
    # Create ellipse mask if requested
    if use_ellipse:
        center_x = crop_width / 2
        center_y = crop_height / 2
        a = (crop_width / 2) * (1 - ellipse_margin)
        b = (crop_height / 2) * (1 - ellipse_margin)
        
        y_coords, x_coords = np.ogrid[:crop_height, :crop_width]
        mask = ((x_coords - center_x) / a) ** 2 + ((y_coords - center_y) / b) ** 2 <= 1
        masked_pixels = img_array[mask]
        
        if len(masked_pixels) == 0:
            return False, 0.0
    else:
        masked_pixels = img_array.reshape(-1, 3)
    
    # Convert target RGB to HSV
    target_hsv = rgb_to_hsv(target_rgb[0], target_rgb[1], target_rgb[2])
    target_h, target_s, target_v = target_hsv
    
    # Convert image pixels to HSV
    hsv = rgb_to_hsv_array(masked_pixels)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    
    # HSV tolerance matching
    # For achromatic colors (white/gray/black with low saturation),
    # hue is meaningless - only check saturation and value
    if target_s < 10:  # Achromatic (white, gray, black)
        # Only check saturation and value
        s_match = np.abs(s - target_s) <= sat_tolerance
        v_match = np.abs(v - target_v) <= val_tolerance
        matches = s_match & v_match
    else:  # Chromatic color - check all three
        # Hue wraps around (0-360), handle wrap-around
        h_diff = np.abs(h - target_h)
        h_diff = np.minimum(h_diff, 360 - h_diff)
        h_match = h_diff <= hue_tolerance
        
        # Saturation and Value are linear
        s_match = np.abs(s - target_s) <= sat_tolerance
        v_match = np.abs(v - target_v) <= val_tolerance
        
        # All three must match
        matches = h_match & s_match & v_match
    
    match_count = np.sum(matches)
    confidence = (match_count / len(hsv)) * 100
    
    result = confidence >= min_confidence
    
    logger.debug(f"HSV tolerance check: target_rgb={target_rgb} (HSV: {target_h:.0f}°, {target_s:.0f}%, {target_v:.0f}%), "
                f"matched={match_count}/{len(hsv)} ({confidence:.1f}%), threshold={min_confidence}%, result={result}")
    
    return result, confidence


def check_target_color(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    target_color: str,
    min_confidence: float = 15.0,
    use_ellipse: bool = True,
    ellipse_margin: float = 0.1
) -> Tuple[bool, float]:
    """
    Check if a target color is present in sufficient quantity.
    
    This is the preferred method for color filtering - instead of finding
    the dominant color, it checks if a SPECIFIC color meets the threshold.
    
    Args:
        image: PIL Image
        bbox: Bounding box (x1, y1, x2, y2)
        target_color: Color to check for ('white', 'red', 'blue', etc.)
        min_confidence: Minimum percentage of pixels needed (default: 15%)
        use_ellipse: If True, only check pixels inside inscribed ellipse (default: True)
        ellipse_margin: Margin as fraction of width/height (default: 0.1 = 10%)
    
    Returns:
        (matches, confidence) - True if color present >= threshold, and percentage
    """
    # Crop region
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(image.width, int(x2)), min(image.height, int(y2))
    
    if x2 <= x1 or y2 <= y1:
        return False, 0.0
    
    crop = image.crop((x1, y1, x2, y2))
    crop_width = x2 - x1
    crop_height = y2 - y1
    
    # Convert to numpy array
    img_array = np.array(crop)
    if len(img_array.shape) != 3 or img_array.shape[2] != 3:
        return False, 0.0
    
    # Create ellipse mask if requested
    mask = None
    if use_ellipse:
        # Ellipse parameters (inscribed with margin)
        center_x = crop_width / 2
        center_y = crop_height / 2
        a = (crop_width / 2) * (1 - ellipse_margin)  # semi-major axis (width)
        b = (crop_height / 2) * (1 - ellipse_margin)  # semi-minor axis (height)
        
        # Create coordinate grid
        y_coords, x_coords = np.ogrid[:crop_height, :crop_width]
        
        # Ellipse equation: ((x-cx)/a)^2 + ((y-cy)/b)^2 <= 1
        mask = ((x_coords - center_x) / a) ** 2 + ((y_coords - center_y) / b) ** 2 <= 1
        
        # Apply mask to image
        masked_pixels = img_array[mask]
        
        if len(masked_pixels) == 0:
            return False, 0.0
    else:
        # Use all pixels
        masked_pixels = img_array.reshape(-1, 3)
    
    # Convert RGB to HSV
    hsv = rgb_to_hsv_array(masked_pixels)
    total_pixels = len(hsv)
    
    # Count pixels matching target color
    color_counts = count_color_pixels(hsv)
    target_count = color_counts.get(target_color, 0)
    confidence = (target_count / total_pixels) * 100
    
    matches = confidence >= min_confidence
    
    # DEBUGGING: Log ALL colors found, not just target
    all_colors_str = ", ".join([f"{c}:{cnt}" for c, cnt in sorted(color_counts.items(), key=lambda x: x[1], reverse=True)])
    mask_info = f" (ellipse: {int(a*2)}x{int(b*2)})" if use_ellipse else ""
    logger.debug(f"Color check{mask_info}: target={target_color}, found={all_colors_str}, {target_color}_count={target_count}/{total_pixels} ({confidence:.1f}%), threshold={min_confidence}%, match={matches}")
    
    return matches, confidence


def count_color_pixels(hsv: np.ndarray) -> dict:
    """
    Count pixels for each color using vectorized operations.
    
    Args:
        hsv: Nx3 array of HSV values (H: 0-360, S/V: 0-100)
        Returns:
            Dict mapping color names to pixel counts
    """
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    
    # Count pixels for each color using vectorized operations
    color_counts = {}
    
    # White: low saturation, high value (relaxed for poor lighting and off-white clothing)
    white_mask = (s < 35) & (v > 55)  # Permissive: s<35 (was 30), v>55 (was 60)
    white_count = np.sum(white_mask)
    if white_count > 0:
        color_counts['white'] = white_count
    
    # Black: low value
    black_mask = (s < 30) & (v < 30)  # Match original threshold
    black_count = np.sum(black_mask)
    if black_count > 0:
        color_counts['black'] = black_count
    
    # Gray: low saturation, mid value (expanded range to capture gray classification better)
    gray_mask = (s < 30) & (v >= 30) & (v <= 60)
    gray_count = np.sum(gray_mask)
    if gray_count > 0:
        color_counts['gray'] = gray_count
    
    # Saturated colors (s >= 25)
    saturated_mask = s >= 25
    if np.any(saturated_mask):
        h_sat = h[saturated_mask]
            # Red (wraps around 0)
    red_mask = ((h_sat >= 0) & (h_sat < 15)) | ((h_sat >= 345) & (h_sat < 360))
    red_count = np.sum(red_mask)
    if red_count > 0:
        color_counts['red'] = red_count
            # Orange
    orange_mask = (h_sat >= 15) & (h_sat < 45)
    if np.sum(orange_mask) > 0:
        color_counts['orange'] = np.sum(orange_mask)
            # Yellow
    yellow_mask = (h_sat >= 45) & (h_sat < 75)
    if np.sum(yellow_mask) > 0:
        color_counts['yellow'] = np.sum(yellow_mask)
            # Green
    green_mask = (h_sat >= 75) & (h_sat < 155)
    if np.sum(green_mask) > 0:
        color_counts['green'] = np.sum(green_mask)
            # Cyan
    cyan_mask = (h_sat >= 155) & (h_sat < 200)
    if np.sum(cyan_mask) > 0:
        color_counts['cyan'] = np.sum(cyan_mask)
            # Blue
    blue_mask = (h_sat >= 200) & (h_sat < 260)
    if np.sum(blue_mask) > 0:
        color_counts['blue'] = np.sum(blue_mask)
            # Purple
    purple_mask = (h_sat >= 260) & (h_sat < 300)
    if np.sum(purple_mask) > 0:
        color_counts['purple'] = np.sum(purple_mask)
            # Pink
    pink_mask = (h_sat >= 300) & (h_sat < 345)
    if np.sum(pink_mask) > 0:
        color_counts['pink'] = np.sum(pink_mask)
    
    return color_counts


def get_color_by_pixel_percentage(
    hsv_pixels: np.ndarray,
    min_percentage: float = MIN_PIXEL_PERCENTAGE
) -> Optional[str]:
    """
    Get color if at least min_percentage of pixels match a color range.
    
    Args:
        hsv_pixels: Nx3 array of HSV values
        min_percentage: Minimum percentage threshold
    
    Returns:
        Color name or None
    """
    total_pixels = len(hsv_pixels)
    
    for color_name, ranges in HSV_COLOR_RANGES.items():
        if not ranges:  # Skip special cases (white/black/gray)
            continue
        matching_pixels = 0
        for h, s, v in hsv_pixels:
            if matches_hsv_range(h, s, v, color_name):
                matching_pixels += 1
        percentage = (matching_pixels / total_pixels) * 100
        if percentage >= min_percentage:
            return color_name
    
    return None


def rgb_to_hsv(r: int, g: int, b: int) -> Tuple[float, float, float]:
    """
    Convert single RGB color to HSV.
    
    Args:
        r, g, b: RGB values (0-255)
    
    Returns:
        Tuple (h, s, v) where H is 0-360, S and V are 0-100
    """
    # Normalize to 0-1
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    max_val = max(r_norm, g_norm, b_norm)
    min_val = min(r_norm, g_norm, b_norm)
    delta = max_val - min_val
    
    # Hue
    if delta == 0:
        h = 0
    elif max_val == r_norm:
        h = (60 * ((g_norm - b_norm) / delta) + 360) % 360
    elif max_val == g_norm:
        h = (60 * ((b_norm - r_norm) / delta) + 120) % 360
    else:  # max_val == b_norm
        h = (60 * ((r_norm - g_norm) / delta) + 240) % 360
    
    # Saturation (0-100)
    s = 0 if max_val == 0 else (delta / max_val) * 100
    
    # Value (0-100)
    v = max_val * 100
    
    return (h, s, v)


def rgb_to_hsv_array(rgb_pixels: np.ndarray) -> np.ndarray:
    """
    Convert RGB pixel array to HSV.
    
    Args:
        rgb_pixels: Nx3 array of RGB values (0-255)
        Returns:
            Nx3 array of HSV values (H: 0-360, S/V: 0-100)
    """
    # Normalize to 0-1
    rgb = rgb_pixels / 255.0
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    
    max_val = np.max(rgb, axis=1)
    min_val = np.min(rgb, axis=1)
    delta = max_val - min_val
    
    # Hue
    h = np.zeros(len(rgb))
    mask = delta != 0
    
    r_max = (max_val == r) & mask
    g_max = (max_val == g) & mask
    b_max = (max_val == b) & mask
    
    h[r_max] = (60 * ((g[r_max] - b[r_max]) / delta[r_max]) + 360) % 360
    h[g_max] = (60 * ((b[g_max] - r[g_max]) / delta[g_max]) + 120) % 360
    h[b_max] = (60 * ((r[b_max] - g[b_max]) / delta[b_max]) + 240) % 360
    
    # Saturation (0-100)
    s = np.zeros(len(rgb))
    s[max_val != 0] = (delta[max_val != 0] / max_val[max_val != 0]) * 100
    
    # Value (0-100)
    v = max_val * 100
    
    return np.column_stack([h, s, v])


def hsv_to_color_name(h: float, s: float, v: float) -> Optional[str]:
    """
    Convert HSV values to color name.
    
    Args:
        h: Hue (0-360 degrees)
        s: Saturation (0-100)
        v: Value/Brightness (0-100)
    
    Returns:
        Color name or None
    """
    # Low saturation = gray/white/black
    # More lenient thresholds for real-world clothing with shadows/folds
    if s < 25:  # Increased from 20 - white clothing can have slight color tint
        if v > 65:  # Decreased from 80 - white in shadow should still be white
            return 'white'
        if v < 30:
            return 'black'
        return 'gray'
    
    # Check hue ranges
    for color_name, ranges in HSV_COLOR_RANGES.items():
        if not ranges:  # Skip special cases
            continue
        for hue_range in ranges:
            if len(hue_range) == 2:
                h_min, h_max = hue_range
                if h_min <= h < h_max:
                    return color_name
    
    return None


def matches_color(
    image: Image.Image,
    bbox: Tuple[int, int, int, int],
    color_name: str
) -> bool:
    """
    Check if region matches specified color.
    
    Args:
        image: PIL Image
    bbox: Bounding box (x1, y1, x2, y2)
    color_name: Target color name
        Returns:
            True if color matches
    """
    extracted_color = extract_dominant_color(image, bbox)
    return extracted_color == color_name


def matches_hsv_range(h: float, s: float, v: float, color_name: str) -> bool:
    """
    Check if HSV values match color range.
    
    Args:
        h: Hue (0-360)
    s: Saturation (0-100)
    v: Value (0-100)
    color_name: Target color
        Returns:
            True if matches
    """
    # Special cases
    if color_name == 'white':
        return s < 20 and v > 80
    if color_name == 'black':
        return v < 30
    if color_name == 'gray':
        return s < 20 and 30 <= v <= 80
    
    # Check saturation/value thresholds
    if s < MIN_SATURATION or v < MIN_VALUE:
        return False
    
    # Check hue ranges
    ranges = HSV_COLOR_RANGES.get(color_name, [])
    for hue_range in ranges:
        if len(hue_range) == 2:
            h_min, h_max = hue_range
            if h_min <= h < h_max:
                return True
    
    return False


def hex_to_color_name(hex_color: str) -> str:
    """Convert hex color to nearest color name.
    
    Args:
        hex_color: Hex color string (e.g., '#FF0000' or 'FF0000')
        Returns:
            Color name (e.g., 'red', 'blue', 'white') or 'unknown'
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Parse RGB
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except (ValueError, IndexError):
        return 'unknown'
    
    # Convert RGB to HSV
    r_norm = r / 255.0
    g_norm = g / 255.0
    b_norm = b / 255.0
    
    max_val = max(r_norm, g_norm, b_norm)
    min_val = min(r_norm, g_norm, b_norm)
    delta = max_val - min_val
    
    # Value (brightness)
    v = max_val * 100
    
    # Saturation
    s = (delta / max_val * 100) if max_val > 0 else 0
    
    # Hue
    if delta == 0:
        h = 0
    elif max_val == r_norm:
        h = 60 * (((g_norm - b_norm) / delta) % 6)
    elif max_val == g_norm:
        h = 60 * (((b_norm - r_norm) / delta) + 2)
    else:
        h = 60 * (((r_norm - g_norm) / delta) + 4)
    
    # Legacy function - no longer used (switched to RGB arrays)
    # Just return unknown since this code path is deprecated
    return 'unknown'
