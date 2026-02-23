"""Detection tracker with deduplication and confirmation."""

import time
import hashlib
import logging
from enum import Enum
from typing import Dict, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

from camera_framework import BaseTask, CocoCategory
from .config import TrackerConfig


logger = logging.getLogger(__name__)


class TrackStateEnum(Enum):
    """Track state machine states."""
    UNKNOWN = "unknown"           # Accumulating confirmations
    PENDING_VLM = "pending_vlm"   # Sent to VLM worker, awaiting yes/no result
    CONFIRMED = "confirmed"       # Confirmed, emitted to downstream
    SUPPRESSED = "suppressed"     # Suppressed during cooldown


@dataclass
class TrackState:
    """State for one tracked detection within one filter."""
    
    # Identification
    track_id: str
    state: TrackStateEnum = TrackStateEnum.UNKNOWN
    
    # Timing
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    confirmation_count: int = 0
    confirmed_at: Optional[float] = None
    
    # Latest (for matching next detection)
    latest_bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)
    latest_attributes: Dict = field(default_factory=dict)
    latest_colors: Dict = field(default_factory=dict)
    latest_category: int = 0
    
    # Best (for alert/VLM)
    best_score: float = 0.0
    best_crop_jpeg: Optional[bytes] = None
    best_attributes: Dict = field(default_factory=dict)
    best_colors: Dict = field(default_factory=dict)
    best_confidence: float = 0.0
    best_timestamp: float = 0.0
    best_bbox: Tuple[float, float, float, float] = (0, 0, 0, 0)

    # VLM retry tracking
    vlm_attempts: int = 0


class DetectionTracker(BaseTask):
    """
    Deduplicates detection stream and manages alert lifecycle.
    
    Per-filter tracking with state machine:
    UNKNOWN → PENDING_VLM → SUPPRESSED  (when vlm_required=True)
    UNKNOWN → SUPPRESSED                 (when vlm_required=False)

    The VLM feedback path calls handle_vlm_result() which either:
    - Confirms the track (SUPPRESSED, cooldown starts)
    - Rejects it (reset to UNKNOWN for retry, up to vlm_max_attempts)
    - Gets an invalid response (reset to UNKNOWN, attempt not counted)    
    Emits confirmed detections to outputs[0] (alert buffer).
    Includes vlm_required flag in event for future VLM routing.
    
    Parameters (tunable):
    - confirmation_threshold: Number of detections needed (default: 5)
    - confirmation_window: Time window in seconds (default: 3.0)
    - cooldown_duration: Suppression time in seconds (default: 120.0)
    - ttl_duration: Time to keep LOST tracks (default: 300.0)
    - iou_threshold: IoU for track matching (default: 0.5)
    - attribute_similarity_threshold: Min similarity (default: 0.7)
    
    Example:
        from .detection_filter import DetectionFilter
        
        filter_config = DetectionFilter()
        tracker = DetectionTracker(filter_config=filter_config)
        
        tracker.add_input("clustered", clustered_buffer)
        tracker.add_output("alerts", alert_buffer)
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
    
    def __init__(
        self,
        config: TrackerConfig,
        name: str = "detection_tracker",
        filter_config: Optional['DetectionFilter'] = None,
        vlm_max_attempts: int = 3,
    ):
        """
        Initialize detection tracker.
        
        Args:
            config: TrackerConfig instance with tracker settings
            name: Task name
            filter_config: Shared DetectionFilter instance
        
        Raises:
            ValueError: If config is None
        """
        super().__init__(name=name)
        
        if config is None:
            raise ValueError("TrackerConfig is required for DetectionTracker")
        
        # Shared filter configuration (thread-safe)
        self.config = config
        self.filter_config = filter_config
        
        # Confirmation parameters
        self.confirmation_threshold = config.confirmation.threshold
        self.confirmation_window = config.confirmation.window
        
        # Matching parameters
        self.iou_threshold = config.matching.iou_threshold
        self.attribute_similarity_threshold = config.matching.attribute_similarity_threshold
        
        # Lifecycle parameters
        self.cooldown_duration = config.lifecycle.cooldown_duration
        self.ttl_duration = config.lifecycle.ttl_duration
        
        # Cropping parameters (stored for image processing)
        self.horizontal_padding = config.cropping.horizontal_padding
        self.vertical_padding = config.cropping.vertical_padding
        self.jpeg_quality = config.cropping.jpeg_quality

        # VLM retry limit
        self.vlm_max_attempts = vlm_max_attempts
        
        # Track storage: filter_tracks[filter_id][track_id] = TrackState
        self.filter_tracks: Dict[str, Dict[str, TrackState]] = {}
        
        # Metrics
        self.stats = {
            'detections_processed': 0,
            'tracks_created': 0,
            'tracks_confirmed': 0,
            'alerts_emitted': 0,
            'tracks_expired': 0,
        }
        
        logger.info(f"DetectionTracker initialized - confirmation_threshold={config.confirmation.threshold}, "
                   f"iou_threshold={config.matching.iou_threshold}")
        logger.debug(f"Config - cooldown={config.lifecycle.cooldown_duration}s, ttl={config.lifecycle.ttl_duration}s, "
                    f"padding=({config.cropping.horizontal_padding}, {config.cropping.vertical_padding})")
    
    def process(self) -> None:
        """Main processing loop - deduplicates detections and emits confirmations."""
        logger.debug(f"Tracker.process() called - inputs={len(self.inputs) if self.inputs else 0}, filter_config={self.filter_config is not None}")
        
        if not self.inputs or not self.filter_config:
            logger.debug(f"Tracker early return: inputs={self.inputs is not None}, filter_config={self.filter_config is not None}")
            return
        
        input_buffer = list(self.inputs.values())[0]
        message = input_buffer.get()
        logger.debug(f"Tracker got message: {message is not None}")
        if not message:
            return
        
        detections = message.get("detections", [])
        
        if detections:
            logger.debug(f"Tracker received {len(detections)} detections from clusterer")
        else:
            logger.debug("Tracker got context but no detections")
        
        for detection in detections:
            # Skip synthetic placeholder detections (created by AttributeColorFilter when
            # all real detections are filtered out, just to keep the image pipeline alive).
            if detection.category == CocoCategory.UNKNOWN:
                logger.debug("Skipping synthetic UNKNOWN detection (image-pipeline placeholder)")
                continue

            self.stats['detections_processed'] += 1
            
            # Debug detection object
            logger.debug(f"Processing detection: type={type(detection)}, has category={hasattr(detection, 'category')}")
            if hasattr(detection, 'category'):
                logger.debug(f"  category type={type(detection.category)}, value={detection.category}")
            
            # Check which filters this detection matches
            matching_filters = self._get_matching_filters(detection)
            logger.debug(f"Detection {detection.category} matched {len(matching_filters)} filters")
            
            for search_filter in matching_filters:
                # Find or create track
                track_id = self._find_matching_track(detection, search_filter.id)
                
                if track_id is None:
                    # Create new track
                    track_id = self._create_track(detection, search_filter)
                
                # Update track
                self._update_track(track_id, detection, search_filter)
        
        # Cleanup expired tracks
        self._cleanup_expired_tracks()
    
    def _get_matching_filters(self, detection) -> list:
        """Get all filters that this detection matches.
        
        Uses pre-matched filter IDs from AttributeColorFilter (stored in metadata).
        Falls back to full re-matching if metadata is missing.
        """
        # Check if detection already has matched filter IDs from AttributeColorFilter
        matched_filter_ids = detection.metadata.get('matched_filters', [])
        
        if matched_filter_ids:
            # Use pre-matched filters - convert IDs to filter objects
            matching = []
            for search_filter in self.filter_config.get_filters():
                if search_filter.id in matched_filter_ids:
                    matching.append(search_filter)
            logger.debug(f"Using pre-matched filters: {len(matching)} matches from {len(matched_filter_ids)} IDs")
            return matching
        
        # Fallback: Full re-matching (shouldn't happen if AttributeColorFilter ran)
        logger.warning("Detection missing matched_filters metadata, re-matching filters")
        matching = []
        
        for search_filter in self.filter_config.get_filters():
            # Check category
            if not search_filter.has_category(detection.category.id):
                continue
            
            # For person, check attribute match
            if detection.category == CocoCategory.PERSON:
                attributes = detection.metadata.get('attributes', {})
                score = self._compute_attribute_score(attributes, search_filter)
                if score < 0.5:  # Require 50% attribute match
                    continue
            
            matching.append(search_filter)
        
        return matching
    
    def _find_matching_track(self, detection, filter_id: str) -> Optional[str]:
        """
        Find existing track that matches this detection.
        
        Matching criteria:
        - IoU > threshold
        - Same category
        - Attribute similarity > threshold (for person)
        - Color similarity (if both have colors)
        
        Returns: track_id or None
        """
        tracks = self._get_tracks_for_filter(filter_id)
        
        for track_id, track_state in tracks.items():
            # Skip if track is too old (hasn't been seen recently)
            if time.time() - track_state.last_seen > self.ttl_duration:
                continue
            
            # Check IoU
            iou = self._compute_iou(detection.bbox, track_state.latest_bbox)
            if iou < self.iou_threshold:
                continue
            
            # Check category
            if detection.category.id != track_state.latest_category:
                continue
            
            # Check attribute similarity (for person)
            if detection.category == CocoCategory.PERSON:
                attrs = detection.metadata.get('attributes', {})
                sim = self._compute_attribute_similarity(attrs, track_state.latest_attributes)
                if sim < self.attribute_similarity_threshold:
                    continue
            
            # Check color similarity
            colors = detection.metadata.get('colors', {})
            if not self._colors_match(colors, track_state.latest_colors):
                continue
            
            # Found match
            return track_id
        
        return None
    
    def _create_track(self, detection, search_filter) -> str:
        """Create new track in UNKNOWN state."""
        track_id = self._generate_track_id(detection)
        
        track_state = TrackState(
            track_id=track_id,
            state=TrackStateEnum.UNKNOWN,
            first_seen=time.time(),
            last_seen=time.time(),
            latest_category=detection.category.id,
        )
        
        # Store in filter tracks
        tracks = self._get_tracks_for_filter(search_filter.id)
        tracks[track_id] = track_state
        self.stats['tracks_created'] += 1
        
        logger.info(f"Created new track {track_id} for filter {search_filter.id}")
        
        return track_id
        tracks[track_id] = track_state
        
        self.stats['tracks_created'] += 1
        logger.debug(f"Created track {track_id} for filter {search_filter.id}")
        
        return track_id
    
    def _update_track(self, track_id: str, detection, search_filter):
        """Update existing track with new detection."""
        tracks = self._get_tracks_for_filter(search_filter.id)
        track_state = tracks[track_id]
        
        # Update latest (for matching next detection)
        track_state.latest_bbox = detection.bbox
        track_state.latest_attributes = detection.metadata.get('attributes', {})
        track_state.latest_colors = detection.metadata.get('colors', {})
        track_state.latest_category = detection.category.id
        track_state.last_seen = time.time()
        
        # Skip if already confirmed/suppressed
        if track_state.state != TrackStateEnum.UNKNOWN:
            return
        
        # Score this detection
        score = self._score_detection(detection, search_filter)
        
        # Update best if this is better
        if score > track_state.best_score:
            track_state.best_score = score
            track_state.best_crop_jpeg = self._crop_and_encode(detection)
            track_state.best_attributes = detection.metadata.get('attributes', {})
            track_state.best_colors = detection.metadata.get('colors', {})
            track_state.best_confidence = detection.confidence
            track_state.best_timestamp = time.time()
            track_state.best_bbox = detection.bbox
        
        # Increment count
        track_state.confirmation_count += 1
        
        # Check for confirmation
        time_window = time.time() - track_state.first_seen
        if (track_state.confirmation_count >= self.confirmation_threshold and
            time_window <= self.confirmation_window):
            self._emit_confirmation(track_state, search_filter)
    
    def _score_detection(self, detection, search_filter) -> float:
        """
        Score detection for best frame selection.
        
        Person: dot product of attributes
        Non-person: color match quality
        Cluster: score based on primary component
        """
        if detection.category == CocoCategory.PERSON:
            # Use attribute dot product
            attributes = detection.metadata.get('attributes', {})
            return self._compute_attribute_score(attributes, search_filter)
        else:
            # Score by color match
            colors = detection.metadata.get('colors', {})
            required_colors = search_filter.color_requirements.get(detection.category.id, [])
            
            if not required_colors:
                return 1.0  # No color requirements
            
            # Count color matches
            score = 0.0
            for region, extracted_color in colors.items():
                if extracted_color in required_colors:
                    score += 1.0
            
            # Normalize by number of required colors
            return score / len(required_colors) if required_colors else 1.0
    
    def _crop_and_encode(self, detection) -> Optional[bytes]:
        """Extract bbox region with padding and encode as JPEG."""
        if not detection.source_image or not CV2_AVAILABLE:
            return None
        
        # Get image as numpy array
        if hasattr(detection.source_image, 'numpy'):
            image = detection.source_image.numpy()
        else:
            image = np.array(detection.source_image)
        
        # Convert RGB to BGR for cv2.imencode (PIL Images are RGB, OpenCV expects BGR)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = image[:, :, ::-1].copy()  # RGB → BGR
        
        # Calculate bbox with padding from config
        x1, y1, x2, y2 = detection.bbox
        width = x2 - x1
        height = y2 - y1
        
        pad_x = width * self.horizontal_padding
        pad_y = height * self.vertical_padding
        
        # Apply padding and clamp to image bounds
        x1 = max(0, int(x1 - pad_x))
        y1 = max(0, int(y1 - pad_y))
        x2 = min(image.shape[1], int(x2 + pad_x))
        y2 = min(image.shape[0], int(y2 + pad_y))
        
        cropped = image[y1:y2, x1:x2]
        
        # Encode to JPEG with quality from config
        success, encoded = cv2.imencode('.jpg', cropped, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not success:
            return None
        
        return encoded.tobytes()
    
    def _emit_confirmation(self, track_state: TrackState, search_filter):
        """Emit confirmation event.

        If the filter requires VLM verification, transitions the track to
        PENDING_VLM and waits for handle_vlm_result() to make the final call.
        Otherwise transitions directly to SUPPRESSED.
        """
        vlm_required = getattr(search_filter, 'vlm_required', False)
        vlm_reasoning = getattr(search_filter, 'vlm_reasoning', '')

        event = {
            'filter_id': search_filter.id,
            'track_id': track_state.track_id,
            'crop_jpeg': track_state.best_crop_jpeg,
            'attributes': track_state.best_attributes,
            'colors': track_state.best_colors,
            'confidence': track_state.best_confidence,
            'timestamp': track_state.best_timestamp,
            'bbox': track_state.best_bbox,
            'confirmation_count': track_state.confirmation_count,
            'first_seen': track_state.first_seen,
            'vlm_required': vlm_required,
            'vlm_reasoning': vlm_reasoning,
        }

        # Send to VLM queue (router will decide if VLM verification is needed)
        if self.outputs:
            output_buffer = list(self.outputs.values())[0]
            output_buffer.put({"confirmations": [event]})
            logger.info(f"Track {track_state.track_id} confirmed for filter {search_filter.id} (vlm_required={vlm_required})")
            self.stats['alerts_emitted'] += 1

        # State transition: wait for VLM feedback if required, suppress immediately otherwise
        if vlm_required:
            track_state.state = TrackStateEnum.PENDING_VLM
        else:
            track_state.state = TrackStateEnum.SUPPRESSED
            track_state.confirmed_at = time.time()

        self.stats['tracks_confirmed'] += 1

    def handle_vlm_result(self, filter_id: str, track_id: str, confirmed: bool, is_valid: bool) -> None:
        """Process VLM verification result and update track state.

        Called by SmolVLMVerifier after inference completes (in-memory callback,
        no queues involved). The track must be in PENDING_VLM state; stale results
        for expired or already-transitioned tracks are silently ignored.

        Args:
            filter_id: Filter the track belongs to
            track_id: Track to update
            confirmed: True if VLM answered yes
            is_valid: True if VLM produced a parseable yes/no response
        """
        tracks = self.filter_tracks.get(filter_id, {})
        track = tracks.get(track_id)

        if track is None:
            logger.debug(f"VLM result for unknown/expired track {track_id} (filter={filter_id}), ignoring")
            return

        if track.state != TrackStateEnum.PENDING_VLM:
            logger.debug(f"Track {track_id} is no longer PENDING_VLM (state={track.state.value}), ignoring")
            return

        if confirmed:
            # VLM said yes — start cooldown
            track.state = TrackStateEnum.SUPPRESSED
            track.confirmed_at = time.time()
            logger.info(f"Track {track_id} VLM-confirmed — entering cooldown")

        elif is_valid:
            # Clean NO — count attempt and maybe retry
            track.vlm_attempts += 1
            if track.vlm_attempts < self.vlm_max_attempts:
                self._reset_track_for_retry(track)
                logger.info(
                    f"Track {track_id} VLM-rejected "
                    f"(attempt {track.vlm_attempts}/{self.vlm_max_attempts}) — reset for retry"
                )
            else:
                track.state = TrackStateEnum.SUPPRESSED
                track.confirmed_at = time.time()
                logger.info(f"Track {track_id} VLM-rejected after {track.vlm_attempts} attempts — suppressed silently")

        else:
            # Invalid/garbled response — retry freely, don't charge an attempt
            self._reset_track_for_retry(track)
            logger.info(
                f"Track {track_id} VLM invalid response "
                f"(attempts so far: {track.vlm_attempts}) — reset for retry"
            )

    def _reset_track_for_retry(self, track: TrackState) -> None:
        """Reset tracking state so the next VLM attempt uses a fresh crop.

        Clears the confirmation window and best-frame selection so that
        detections accumulated after the reset produce a new crop instead
        of re-sending the already-rejected image.
        """
        track.state = TrackStateEnum.UNKNOWN
        track.confirmation_count = 0
        track.first_seen = time.time()
        # Clear best-frame so _update_track captures a fresh crop on retry
        track.best_score = 0.0
        track.best_crop_jpeg = None
        track.best_attributes = {}
        track.best_colors = {}
        track.best_confidence = 0.0
        track.best_timestamp = 0.0
    
    def _cleanup_expired_tracks(self):
        """Remove tracks that haven't been seen recently."""
        current_time = time.time()
        
        for filter_id, tracks in list(self.filter_tracks.items()):
            expired = []
            
            for track_id, track_state in tracks.items():
                # Check TTL
                if current_time - track_state.last_seen > self.ttl_duration:
                    expired.append(track_id)
                    continue
                
                # Check cooldown expiration for suppressed tracks
                if (track_state.state == TrackStateEnum.SUPPRESSED and
                    track_state.confirmed_at and
                    current_time - track_state.confirmed_at > self.cooldown_duration):
                    expired.append(track_id)
            
            # Remove expired
            for track_id in expired:
                del tracks[track_id]
                self.stats['tracks_expired'] += 1
                logger.debug(f"Expired track {track_id} from filter {filter_id}")
    
    # ==================== Utility Methods ====================
    
    def _get_tracks_for_filter(self, filter_id: str) -> Dict[str, TrackState]:
        """Get or create track dict for filter."""
        if filter_id not in self.filter_tracks:
            self.filter_tracks[filter_id] = {}
        return self.filter_tracks[filter_id]
    
    def _generate_track_id(self, detection) -> str:
        """Generate unique track ID."""
        # Hash of category, bbox, and timestamp
        data = f"{detection.category.id}_{detection.bbox}_{time.time()}"
        return hashlib.md5(data.encode()).hexdigest()[:16]
    
    def _compute_iou(self, bbox1: Tuple, bbox2: Tuple) -> float:
        """Calculate intersection over union."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _compute_attribute_similarity(self, attrs1: Dict, attrs2: Dict) -> float:
        """Compute dot product similarity of attribute vectors."""
        if not attrs1 or not attrs2:
            return 0.0
        
        # Build vectors
        vec1 = []
        vec2 = []
        
        for attr_name in self.PA100K_ATTRIBUTES:
            val1 = 1 if attrs1.get(attr_name, {}).get('value', False) else 0
            val2 = 1 if attrs2.get(attr_name, {}).get('value', False) else 0
            vec1.append(val1)
            vec2.append(val2)
        
        # Dot product
        dot_product = sum(v1 * v2 for v1, v2 in zip(vec1, vec2))
        
        # Normalize by L2 norms
        norm1 = sum(v * v for v in vec1) ** 0.5
        norm2 = sum(v * v for v in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _compute_attribute_score(self, attributes: Dict, search_filter) -> float:
        """Compute dot product score against filter requirements."""
        # Convert attribute_mask to set of required attribute names
        filter_attrs = set()
        for i, enabled in enumerate(search_filter.attribute_mask):
            if enabled and i < len(self.PA100K_ATTRIBUTES):
                filter_attrs.add(self.PA100K_ATTRIBUTES[i])
        
        if not filter_attrs:
            return 1.0
        
        # Build detection vector
        det_vector = []
        for attr_name in self.PA100K_ATTRIBUTES:
            attr_data = attributes.get(attr_name, {})
            value = 1 if attr_data.get('value', False) else 0
            det_vector.append(value)
        
        # Build filter mask
        filter_mask = []
        for attr_name in self.PA100K_ATTRIBUTES:
            filter_mask.append(1 if attr_name in filter_attrs else 0)
        
        # Dot product
        dot_product = sum(d * f for d, f in zip(det_vector, filter_mask))
        
        # Normalize
        num_filter_attrs = len(filter_attrs)
        return dot_product / num_filter_attrs if num_filter_attrs > 0 else 1.0
    
    def _colors_match(self, colors1: Dict, colors2: Dict) -> bool:
        """Check if color dicts have any overlap."""
        if not colors1 or not colors2:
            return True  # No colors to compare
        
        # Check if any color values match
        set1 = set(colors1.values())
        set2 = set(colors2.values())
        
        return bool(set1 & set2)  # Any intersection
    
    def _matches_color_requirements(self, detection, search_filter) -> bool:
        """Check if detection meets filter color requirements."""
        color_reqs = search_filter.color_requirements.get(detection.category.id, [])
        if not color_reqs:
            return True
        
        extracted_colors = detection.metadata.get('colors', {})
        if not extracted_colors:
            return False
        
        # Check if any extracted color matches
        for color_name in color_reqs:
            for region, extracted_color in extracted_colors.items():
                if extracted_color == color_name:
                    return True
        
        return False
    
    def get_stats(self) -> dict:
        """Get tracker statistics."""
        total_tracks = sum(len(tracks) for tracks in self.filter_tracks.values())
        
        return {
            **self.stats,
            'active_tracks': total_tracks,
            'active_filters': len(self.filter_tracks),
        }
