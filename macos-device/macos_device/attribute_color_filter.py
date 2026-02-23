"""Attribute and color filter - extracts colors and filters detections."""

import logging
from typing import Optional, Dict
from camera_framework import BaseTask, CocoCategory
from camera_framework.detection import Detection, ImageFormat
from .color_utils import extract_dominant_color, check_target_rgb_simple
from .config import ColorFilterConfig

logger = logging.getLogger(__name__)


class AttributeColorFilter(BaseTask):
    """Filters detections using attributes and color requirements.
    
    Extracts colors from detection regions based on category and attributes,
    then filters detections that match the search criteria.
    
    Color extraction strategy:
    - Person with attributes: Extract from attribute-specific regions
      (e.g., Hat -> top 15% of bbox, UpperStride -> middle 60%, etc.)
    - Person without attributes: Extract from full bbox
    - Non-person objects: Extract from full bbox
    - Clusters (parent with children): Extract from merged bbox
    
    Thread-safe for MQTT updates via shared filter_config.
    
    Example:
        from .detection_filter import DetectionFilter
        from .config import ColorFilterConfig
        
        filter_config = DetectionFilter()
        color_config = ColorFilterConfig.load("color_config.yaml")
        
        attr_color_filter = AttributeColorFilter(
            config=color_config,
            filter_config=filter_config
        )
        attr_color_filter.add_input("detections", enriched_buffer)
        attr_color_filter.add_output("filtered", mqtt_buffer)
    """
    
    # Body region categorization for color filtering
    UPPER_BODY_REGIONS = {
        'Hat', 'Glasses', 'UpperStride', 'UpperLogo', 'UpperPlaid', 
        'UpperSplice', 'ShortSleeve', 'LongSleeve', 'LongCoat'
    }
    
    LOWER_BODY_REGIONS = {
        'LowerStripe', 'LowerPattern', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots'
    }
    
    def __init__(
        self,
        config: ColorFilterConfig,
        name: str = "attribute_color_filter",
        filter_config: Optional['DetectionFilter'] = None
    ):
        """
        Initialize attribute and color filter.
        
        Args:
            config: ColorFilterConfig instance with region and matching settings
            name: Task name
            filter_config: Shared DetectionFilter instance
        
        Raises:
            ValueError: If config is None
        """
        super().__init__(name=name)
        
        if config is None:
            raise ValueError("ColorFilterConfig is required for AttributeColorFilter")
        
        # Shared filter configuration (thread-safe)
        self.config = config
        self.filter_config = filter_config
        
        # Build region dictionaries from config
        self.IMAGE_REGIONS = {
            'top': (config.regions.image.top.start, config.regions.image.top.end),
            'middle-top': (config.regions.image.middle_top.start, config.regions.image.middle_top.end),
            'middle-bottom': (config.regions.image.middle_bottom.start, config.regions.image.middle_bottom.end),
            'bottom': (config.regions.image.bottom.start, config.regions.image.bottom.end),
        }
        
        self.PERSON_REGIONS = {
            'Hat': (config.regions.person.hat.start, config.regions.person.hat.end),
            'Glasses': (config.regions.person.glasses.start, config.regions.person.glasses.end),
            'UpperStride': (config.regions.person.upper_stride.start, config.regions.person.upper_stride.end),
            'UpperLogo': (config.regions.person.upper_logo.start, config.regions.person.upper_logo.end),
            'UpperPlaid': (config.regions.person.upper_plaid.start, config.regions.person.upper_plaid.end),
            'UpperSplice': (config.regions.person.upper_splice.start, config.regions.person.upper_splice.end),
            'ShortSleeve': (config.regions.person.short_sleeve.start, config.regions.person.short_sleeve.end),
            'LongSleeve': (config.regions.person.long_sleeve.start, config.regions.person.long_sleeve.end),
            'LongCoat': (config.regions.person.long_coat.start, config.regions.person.long_coat.end),
            'LowerStripe': (config.regions.person.lower_stripe.start, config.regions.person.lower_stripe.end),
            'LowerPattern': (config.regions.person.lower_pattern.start, config.regions.person.lower_pattern.end),
            'Trousers': (config.regions.person.trousers.start, config.regions.person.trousers.end),
            'Shorts': (config.regions.person.shorts.start, config.regions.person.shorts.end),
            'SkirtDress': (config.regions.person.skirt_dress.start, config.regions.person.skirt_dress.end),
            'boots': (config.regions.person.boots.start, config.regions.person.boots.end),
        }
        
        # Store matching thresholds
        self.max_color_diff = config.matching.max_color_diff
        self.min_brightness = config.matching.min_brightness
        self.min_confidence = config.matching.min_confidence
        
        # Store HSV thresholds
        self.white_saturation_max = config.hsv.white_saturation_max
        self.black_value_max = config.hsv.black_value_max
        
        logger.info(f"AttributeColorFilter initialized - max_color_diff={config.matching.max_color_diff}, "
                   f"min_confidence={config.matching.min_confidence}")
        logger.debug(f"HSV - white_sat_max={config.hsv.white_saturation_max}, black_val_max={config.hsv.black_value_max}")
    
    def process(self) -> None:
        """Extract colors and filter detections."""
        if not self.inputs or not self.filter_config:
            logger.warning(f"AttributeColorFilter: inputs={bool(self.inputs)}, filter_config={bool(self.filter_config)}")
            return
        
        input_buffer = list(self.inputs.values())[0]
        message = input_buffer.get()
        if not message:
            logger.debug("AttributeColorFilter: No message available")
            return
        
        detections = message.get("detections", [])
        active_filters = self.filter_config.get_filters()
        logger.debug(f"AttributeColorFilter: {len(detections)} detections, {len(active_filters)} filters")
        
        # Filter detections by attributes and colors
        filtered = []
        for i, d in enumerate(detections):
            matched_filter_ids = self._matches_filter(d)
            if matched_filter_ids:
                # Attach matched filter IDs to detection metadata for tracker
                d.metadata['matched_filters'] = matched_filter_ids
                filtered.append(d)
                logger.debug(f"Det#{i+1} {d.category.label}: ✓ KEPT (filters: {matched_filter_ids})")
            else:
                logger.debug(f"Det#{i+1} {d.category.label}: ✗ REJECTED")
        logger.debug(f"Result: {len(filtered)}/{len(detections)} kept")
        
        # If all detections filtered out, create synthetic full-frame detection
        # This ensures downstream buffers (especially snapshot) always have an image
        if not filtered and detections:
            # Get source image from first detection
            source_detection = detections[0]
            if source_detection.source_image:
                width = source_detection.source_image.width if hasattr(source_detection.source_image, 'width') else source_detection.source_image.size[0]
                height = source_detection.source_image.height if hasattr(source_detection.source_image, 'height') else source_detection.source_image.size[1]
                synthetic = Detection(
                    bbox=(0.0, 0.0, float(width), float(height)),
                    confidence=0.0,  # Low confidence ensures no alerts
                    category=CocoCategory.UNKNOWN,
                    source_image=source_detection.source_image,
                    source_format=source_detection.source_format or ImageFormat.PIL
                )
                filtered.append(synthetic)
        
        # Write to output if any detections (including synthetic)
        if filtered and self.outputs:
            output_buffer = list(self.outputs.values())[0]
            message = {"detections": filtered}
            output_buffer.put(message)
    
    def _extract_and_store_colors(self, detection) -> None:
        """Extract colors from detection and store in metadata.
        
        Args:
            detection: Detection object with source_image and bbox
        """
        if not detection.source_image:
            return
        
        colors = {}
        
        if detection.category == CocoCategory.PERSON:
            # Person - extract colors from attribute regions
            attributes = detection.metadata.get('attributes', {})
            
            for attr_name, attr_data in attributes.items():
                # Only extract color if attribute is present (True)
                if not attr_data.get('value', False):
                    continue
                
                # Get region for this attribute
                region = self.PERSON_REGIONS.get(attr_name)
                if not region:
                    continue
                
                # Extract color from region
                color_bbox = self._get_region_bbox(detection.bbox, region)
                color = extract_dominant_color(
                    detection.source_image,
                    color_bbox
                )
                
                if color:
                    colors[attr_name] = color
        
        else:
            # Non-person object - extract from full bbox
            # Check if this is a cluster (has children)
            bbox = detection.bbox
            if hasattr(detection, 'children') and detection.children:
                # Cluster - use merged bbox
                bbox = self._get_merged_bbox(detection)
            
            color = extract_dominant_color(
                detection.source_image,
                bbox
            )
            
            if color:
                colors['object'] = color
        
        # Store colors in metadata
        if colors:
            if 'colors' not in detection.metadata:
                detection.metadata['colors'] = {}
            detection.metadata['colors'].update(colors)
    
    def _get_region_bbox(self, bbox: tuple, region: tuple) -> tuple:
        """Extract sub-region from bbox.
        
        Args:
            bbox: (x1, y1, x2, y2)
            region: (y_start_pct, y_end_pct) as 0.0-1.0
            
        Returns:
            (x1, y1_new, x2, y2_new)
        """
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        
        y_start_pct, y_end_pct = region
        y1_new = y1 + (height * y_start_pct)
        y2_new = y1 + (height * y_end_pct)
        
        return (x1, y1_new, x2, y2_new)
    
    def _get_merged_bbox(self, detection) -> tuple:
        """Get merged bbox for cluster (parent + all children).
        
        Args:
            detection: Parent detection with children
            
        Returns:
            Merged bbox (x1, y1, x2, y2)
        """
        x1, y1, x2, y2 = detection.bbox
        
        # Expand to include all children
        for child in detection.children:
            cx1, cy1, cx2, cy2 = child.bbox
            x1 = min(x1, cx1)
            y1 = min(y1, cy1)
            x2 = max(x2, cx2)
            y2 = max(y2, cy2)
        
        return (x1, y1, x2, y2)
    
    def _matches_filter(self, detection) -> list:
        """Check if detection matches filter criteria.
        
        Uses dot product scoring for person attributes against filter requirements.
        
        Args:
            detection: Detection to check
            
        Returns:
            List of filter IDs that this detection matches (empty if no match)
        """
        # Non-person detections - check category only (no attributes)
        if detection.category != CocoCategory.PERSON:
            matching_ids = []
            for search_filter in self.filter_config.get_filters():
                if search_filter.has_category(detection.category.id):
                    if self._matches_color_requirements_for_filter(detection, search_filter):
                        matching_ids.append(search_filter.id)
            return matching_ids
        
        # Person detection - use dot product scoring
        attributes = detection.metadata.get('attributes', {})
        if not attributes:
            logger.warning(f"Person has no attributes!")
            return []
        
        active_filters = self.filter_config.get_filters()
        if not active_filters:
            return []
        
        # Check each filter with dot product scoring and collect matches
        matching_ids = []
        for search_filter in active_filters:
            if not search_filter.has_category(CocoCategory.PERSON.id):
                continue
            
            score = self._compute_attribute_score(attributes, search_filter)
            
            # Require strong attribute match (>= 0.75)
            if score < 0.75:
                continue
            
            logger.debug(f"Filter '{search_filter.name}' score={score:.2f}, checking colors...")
            
            # Check color requirements for this filter
            if self._matches_color_requirements_for_filter(detection, search_filter):
                matching_ids.append(search_filter.id)
        
        return matching_ids
    
    def _compute_attribute_score(self, attributes: Dict, search_filter: 'SearchFilter') -> float:
        """Compute dot product score of detection attributes against filter requirements.
        
        Same algorithm as Clusterer._attribute_filter_match():
        - Build detection vector (26 elements, 0/1 for each PA-100K attribute)
        - Build filter mask (1 for required attributes, 0 for others)
        - Compute dot product and normalize by number of filter requirements
        
        Args:
            attributes: Detection attributes dict {attr_name: {'value': bool}}
            search_filter: Filter with attribute_mask boolean array
            
        Returns:
            Score from 0.0 (no match) to 1.0 (perfect match)
        """
        # PA-100K attributes in fixed order (26 total)
        PA100K_ATTRIBUTES = [
            'Female', 'AgeOver60', 'Age18-60', 'AgeLess18',
            'Front', 'Side', 'Back',
            'Hat', 'Glasses',
            'HandBag', 'ShoulderBag', 'Backpack', 'HoldObjectsInFront',
            'ShortSleeve', 'LongSleeve', 'UpperStride', 'UpperLogo', 'UpperPlaid', 'UpperSplice',
            'LowerStripe', 'LowerPattern', 'LongCoat', 'Trousers', 'Shorts', 'Skirt&Dress', 'boots'
        ]
        
        # Convert attribute_mask to set of required attribute names
        filter_attrs = set()
        for i, enabled in enumerate(search_filter.attribute_mask):
            if enabled and i < len(PA100K_ATTRIBUTES):
                filter_attrs.add(PA100K_ATTRIBUTES[i])
        
        if not filter_attrs:
            # No attribute requirements - perfect match
            logger.debug(f"  No attribute requirements in filter, score=1.0")
            return 1.0
        
        logger.debug(f"  Filter requires: {filter_attrs}")
        
        # Build detection attribute vector (0/1 for each PA-100K attribute)
        det_vector = []
        det_present = []
        for attr_name in PA100K_ATTRIBUTES:
            attr_data = attributes.get(attr_name, {})
            value = 1 if attr_data.get('value', False) else 0
            det_vector.append(value)
            if value:
                det_present.append(attr_name)
        
        logger.debug(f"  Detection has: {det_present}")
        
        # Build filter mask (1 for required attributes, 0 for others)
        filter_mask = []
        for attr_name in PA100K_ATTRIBUTES:
            filter_mask.append(1 if attr_name in filter_attrs else 0)
        
        # Compute dot product
        dot_product = sum(d * f for d, f in zip(det_vector, filter_mask))
        
        # Normalize by number of filter requirements
        num_filter_attrs = len(filter_attrs)
        normalized_score = dot_product / num_filter_attrs if num_filter_attrs > 0 else 1.0
        
        logger.debug(f"  Dot product: {dot_product}/{num_filter_attrs} = {normalized_score:.2f}")
        
        return normalized_score
    
    def _matches_color_requirements_for_filter(self, detection, search_filter) -> bool:
        """Check if detection meets color requirements for specific filter.
        
        For persons:
          - Check attribute match (already done before this is called)
          - If attribute has color requirement, check color match - reject if no match
          - If attribute has no color requirement, pass
        
        For non-persons:
          - If category has color requirement, check color match - reject if no match
          - If category has no color requirement, pass
        
        Args:
            detection: Detection with source_image and attributes
            search_filter: Specific filter to check against
            
        Returns:
            True if color requirements met for this filter
        """
        if not detection.source_image:
            logger.warning(f"  No source image for color checking")
            return False
        
        if detection.category == CocoCategory.PERSON:
            # Person: check attribute-specific colors
            attributes = detection.metadata.get('attributes', {})
            
            attr_color_reqs = search_filter.get_attribute_color_requirements()
            
            # Debug: log filter details
            logger.debug(f"  Filter '{search_filter.name}': attribute_colors={search_filter.attribute_colors}")
            logger.debug(f"  Filter '{search_filter.name}': attr_color_reqs={attr_color_reqs}")
            logger.debug(f"  Filter '{search_filter.name}': category color_requirements={search_filter.color_requirements}")
            
            if not attr_color_reqs:
                # No attribute-specific colors - check region-based colors from Groq
                # e.g., {0: {"middle-top": [[255,255,255]]}} means check 25-60% region for white RGB
                region_colors = search_filter.color_requirements.get(CocoCategory.PERSON.id, {})
                if not region_colors:
                    # No color requirements at all - pass
                    logger.debug(f"  No color requirements (attribute or region-based) - passing")
                    return True
                
                # Check region-based colors using HSV tolerance
                logger.debug(f"  Checking region-based RGB colors: {region_colors}")
                
                for region_name, rgb_list in region_colors.items():
                    # Get region bbox from IMAGE_REGIONS
                    region_range = self.IMAGE_REGIONS.get(region_name)
                    if not region_range:
                        logger.warning(f"  Unknown region '{region_name}' - skipping")
                        continue
                    
                    color_bbox = self._get_region_bbox(detection.bbox, region_range)
                    logger.debug(f"  DEBUG: Checking region '{region_name}' ({region_range[0]*100:.0f}%-{region_range[1]*100:.0f}% of bbox)")
                    logger.debug(f"  DEBUG: Detection bbox: {detection.bbox}")
                    logger.debug(f"  DEBUG: Color check bbox (region): {color_bbox}")
                    logger.debug(f"  DEBUG: Source image size: {detection.source_image.size if hasattr(detection.source_image, 'size') else 'unknown'}")
                    
                    for rgb in rgb_list:
                        matches, confidence = check_target_rgb_simple(
                            detection.source_image,
                            color_bbox,
                            tuple(rgb),
                            max_color_diff=self.max_color_diff,
                            min_brightness=self.min_brightness,
                            min_confidence=self.min_confidence
                        )
                        
                        if matches:
                            logger.debug(f"  ✓ RGB {rgb} found in {region_name} @ {confidence:.1f}%")
                            return True
                
                logger.debug(f"  ✗ Region RGB colors not found")
                return False
            
            # Has color requirements - ALL must match
            logger.debug(f"  Checking attribute colors: {attr_color_reqs}")
            all_matched = True
            
            for attr_name, required_colors in attr_color_reqs.items():
                # Detection must have this attribute (already verified by PA-100K scoring)
                if not attributes.get(attr_name, {}).get('value', False):
                    logger.debug(f"  ✗ Missing attribute {attr_name}")
                    all_matched = False
                    break
                
                # Get region for this attribute
                region = self.PERSON_REGIONS.get(attr_name)
                if not region:
                    logger.debug(f"  ✗ No region for {attr_name}")
                    all_matched = False
                    break
                
                # Check color in this region
                color_bbox = self._get_region_bbox(detection.bbox, region)
                matched_color = False
                
                for required_color in required_colors:
                    matches, confidence = check_target_color(
                        detection.source_image,
                        color_bbox,
                        required_color,
                        min_confidence=50.0,
                        use_ellipse=True,
                        ellipse_margin=0.05
                    )
                    
                    if matches:
                        logger.debug(f"  ✓ {required_color} in {attr_name} @ {confidence:.1f}%")
                        matched_color = True
                        break
                
                if not matched_color:
                    logger.debug(f"  ✗ No {required_colors} in {attr_name}")
                    all_matched = False
                    break
            
            return all_matched
        
        else:
            # Non-person: check category-level colors from this specific filter
            region_colors = search_filter.color_requirements.get(detection.category.id, {})
            
            if not region_colors:
                # No color requirements - pass
                logger.debug(f"  No category color requirements - passing")
                return True
            
            # Check region-based colors
            logger.debug(f"  Checking region-based RGB colors: {region_colors}")
            
            for region_name, rgb_list in region_colors.items():
                region_range = self.IMAGE_REGIONS.get(region_name)
                if not region_range:
                    logger.warning(f"  Unknown region '{region_name}' - skipping")
                    continue
                
                color_bbox = self._get_region_bbox(detection.bbox, region_range)
                
                for rgb in rgb_list:
                    matches, confidence = check_target_rgb_simple(
                        detection.source_image,
                        color_bbox,
                        tuple(rgb),
                        max_color_diff=self.max_color_diff,
                        min_brightness=self.min_brightness,
                        min_confidence=self.min_confidence
                    )
                    
                    if matches:
                        logger.debug(f"  ✓ RGB {rgb} found in {region_name} @ {confidence:.1f}%")
                        return True
            
            logger.debug(f"  ✗ Region RGB colors not found")
            return False
