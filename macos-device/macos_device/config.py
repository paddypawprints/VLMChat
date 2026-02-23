"""macOS Device configuration (device-specific tasks and sinks)."""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from pathlib import Path
import yaml


@dataclass
class YoloConfig:
    """YOLO object detection configuration."""
    model_path: str = "yolov8n.pt"
    confidence: float = 0.25
    iou: float = 0.45
    device: str = "cpu"


@dataclass
class AttributesConfig:
    """Person attributes configuration."""
    model_path: str = "/Users/patrick/Downloads/pa_model_best_v3.onnx"
    confidence_threshold: float = 0.5
    batch_size: int = 1


@dataclass
class RegionRange:
    """A vertical region range (start, end) as percentages."""
    start: float
    end: float


@dataclass
class ImageRegionsConfig:
    """Image regions for color extraction (vertical percentages)."""
    top: RegionRange = None
    middle_top: RegionRange = None
    middle_bottom: RegionRange = None
    bottom: RegionRange = None
    
    def __post_init__(self):
        if self.top is None:
            self.top = RegionRange(0.0, 0.25)
        if self.middle_top is None:
            self.middle_top = RegionRange(0.25, 0.60)
        if self.middle_bottom is None:
            self.middle_bottom = RegionRange(0.40, 0.75)
        if self.bottom is None:
            self.bottom = RegionRange(0.75, 1.0)


@dataclass
class PersonRegionsConfig:
    """Person attribute regions for color extraction (vertical percentages)."""
    hat: RegionRange = None
    glasses: RegionRange = None
    upper_stride: RegionRange = None
    upper_logo: RegionRange = None
    upper_plaid: RegionRange = None
    upper_splice: RegionRange = None
    short_sleeve: RegionRange = None
    long_sleeve: RegionRange = None
    long_coat: RegionRange = None
    lower_stripe: RegionRange = None
    lower_pattern: RegionRange = None
    trousers: RegionRange = None
    shorts: RegionRange = None
    skirt_dress: RegionRange = None
    boots: RegionRange = None
    
    def __post_init__(self):
        if self.hat is None:
            self.hat = RegionRange(0.0, 0.15)
        if self.glasses is None:
            self.glasses = RegionRange(0.1, 0.3)
        if self.upper_stride is None:
            self.upper_stride = RegionRange(0.25, 0.65)
        if self.upper_logo is None:
            self.upper_logo = RegionRange(0.25, 0.65)
        if self.upper_plaid is None:
            self.upper_plaid = RegionRange(0.25, 0.65)
        if self.upper_splice is None:
            self.upper_splice = RegionRange(0.25, 0.65)
        if self.short_sleeve is None:
            self.short_sleeve = RegionRange(0.25, 0.5)
        if self.long_sleeve is None:
            self.long_sleeve = RegionRange(0.25, 0.65)
        if self.long_coat is None:
            self.long_coat = RegionRange(0.2, 0.8)
        if self.lower_stripe is None:
            self.lower_stripe = RegionRange(0.5, 0.95)
        if self.lower_pattern is None:
            self.lower_pattern = RegionRange(0.5, 0.95)
        if self.trousers is None:
            self.trousers = RegionRange(0.5, 0.95)
        if self.shorts is None:
            self.shorts = RegionRange(0.5, 0.8)
        if self.skirt_dress is None:
            self.skirt_dress = RegionRange(0.3, 0.95)
        if self.boots is None:
            self.boots = RegionRange(0.8, 1.0)


@dataclass
class ColorRegionsConfig:
    """Color extraction regions configuration."""
    image: ImageRegionsConfig = None
    person: PersonRegionsConfig = None
    
    def __post_init__(self):
        if self.image is None:
            self.image = ImageRegionsConfig()
        if self.person is None:
            self.person = PersonRegionsConfig()


@dataclass
class ColorMatchingConfig:
    """Color matching thresholds configuration."""
    # Attribute matching
    attribute_match_threshold: float = 0.75
    
    # RGB matching (achromatic colors)
    min_confidence: float = 20.0  # Main confidence threshold
    brightness_tolerance: int = 100
    min_brightness: int = 120
    max_color_diff: int = 40
    use_ellipse: bool = True
    ellipse_margin: float = 0.05
    
    # Named color matching
    min_confidence_named: float = 50.0
    hue_tolerance: float = 20.0
    sat_tolerance: float = 50.0
    val_tolerance: float = 50.0


@dataclass
class HSVConfig:
    """HSV color space configuration."""
    min_saturation: int = 50
    min_value: int = 50
    min_pixel_percentage: float = 5.0
    white_saturation_max: int = 50  # Max saturation for achromatic white
    black_value_max: int = 50  # Max value for achromatic black


@dataclass
class ColorFilterConfig:
    """Color filter configuration."""
    regions: ColorRegionsConfig = None
    matching: ColorMatchingConfig = None
    hsv: HSVConfig = None
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = ColorRegionsConfig()
        if self.matching is None:
            self.matching = ColorMatchingConfig()
        if self.hsv is None:
            self.hsv = HSVConfig()


@dataclass
class ClustererWeightsConfig:
    """Clusterer similarity weights."""
    proximity: float = 1.0
    size: float = 1.0
    category: float = 1.5
    attribute: float = 0.8


@dataclass
class ClustererConfig:
    """Clusterer configuration."""
    max_clusters: int = 10
    merge_threshold: float = 0.6
    weights: ClustererWeightsConfig = None
    
    def __post_init__(self):
        if self.weights is None:
            self.weights = ClustererWeightsConfig()


@dataclass
class TrackerConfirmationConfig:
    """Tracker confirmation settings."""
    threshold: int = 2
    window: float = 3.0


@dataclass
class TrackerMatchingConfig:
    """Tracker matching thresholds."""
    iou_threshold: float = 0.5
    attribute_similarity_threshold: float = 0.7


@dataclass
class TrackerCroppingConfig:
    """Tracker image cropping settings."""
    horizontal_padding: float = 0.3   # fraction of box width added to each side
    vertical_padding: float = 0.3     # fraction of box height added top and bottom
    jpeg_quality: int = 95


@dataclass
class TrackerLifecycleConfig:
    """Tracker lifecycle settings."""
    cooldown_duration: float = 120.0
    ttl_duration: float = 300.0


@dataclass
class TrackerConfig:
    """Detection tracker configuration."""
    confirmation: TrackerConfirmationConfig = None
    matching: TrackerMatchingConfig = None
    cropping: TrackerCroppingConfig = None
    lifecycle: TrackerLifecycleConfig = None
    
    def __post_init__(self):
        if self.confirmation is None:
            self.confirmation = TrackerConfirmationConfig()
        if self.matching is None:
            self.matching = TrackerMatchingConfig()
        if self.cropping is None:
            self.cropping = TrackerCroppingConfig()
        if self.lifecycle is None:
            self.lifecycle = TrackerLifecycleConfig()


@dataclass
class SmolVLMConfig:
    """SmolVLM vision-language model configuration."""
    model_path: str = "/Users/patrick/Dev/VLMChat/macos-device/onnx/SmolVLM2-256M-Instruct"
    model_size: str = "256M"  # "256M" or "500M"
    max_new_tokens: int = 10  # yes/no answers need very few tokens
    device_id: str = "mac-dev-01"
    vlm_max_attempts: int = 3       # max NO responses before suppressing a track
    vlm_retry_delay: float = 2.0    # seconds to wait before retrying after INVALID response


@dataclass
class TasksConfig:
    """All task configurations."""
    yolo: YoloConfig = None
    attributes: AttributesConfig = None
    color_filter: ColorFilterConfig = None
    clusterer: ClustererConfig = None
    tracker: TrackerConfig = None
    smolvlm: Optional[SmolVLMConfig] = None
    
    def __post_init__(self):
        if self.yolo is None:
            self.yolo = YoloConfig()
        if self.attributes is None:
            self.attributes = AttributesConfig()
        if self.color_filter is None:
            self.color_filter = ColorFilterConfig()
        if self.clusterer is None:
            self.clusterer = ClustererConfig()
        if self.tracker is None:
            self.tracker = TrackerConfig()
        # smolvlm is optional - only create if in config


@dataclass
class MQTTConfig:
    """MQTT broker configuration."""
    broker_host: str = "localhost"
    broker_port: int = 1883
    device_id: str = "mac-dev-01"
    device_type: str = "macos"
    schemas_path: Optional[str] = None


@dataclass
class SinksConfig:
    """Sink configurations."""
    mqtt: MQTTConfig = None
    
    def __post_init__(self):
        if self.mqtt is None:
            self.mqtt = MQTTConfig()


@dataclass
class MacOSDeviceConfig:
    """Top-level macOS device configuration."""
    tasks: TasksConfig = None
    sinks: SinksConfig = None
    
    def __post_init__(self):
        if self.tasks is None:
            self.tasks = TasksConfig()
        if self.sinks is None:
            self.sinks = SinksConfig()
    
    @classmethod
    def load(cls, path: str) -> 'MacOSDeviceConfig':
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            MacOSDeviceConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        
        with open(config_path) as f:
            data = yaml.safe_load(f)
        
        if not data:
            raise ValueError(f"Empty config file: {path}")
        
        # Parse tasks
        tasks_data = data.get('tasks', {})
        yolo = YoloConfig(**tasks_data.get('yolo', {}))
        attributes = AttributesConfig(**tasks_data.get('attributes', {}))
        
        color_filter_data = tasks_data.get('color_filter', {})
        
        # Parse regions
        regions_data = color_filter_data.get('regions', {})
        image_regions_data = regions_data.get('image', {})
        person_regions_data = regions_data.get('person', {})
        
        # Helper to parse RegionRange from tuple or dict
        def parse_region(data, default_start, default_end):
            if isinstance(data, (list, tuple)):
                return RegionRange(data[0], data[1])
            elif isinstance(data, dict):
                return RegionRange(data.get('start', default_start), data.get('end', default_end))
            else:
                return RegionRange(default_start, default_end)
        
        # Parse image regions
        image_regions = ImageRegionsConfig(
            top=parse_region(image_regions_data.get('top'), 0.0, 0.25),
            middle_top=parse_region(image_regions_data.get('middle_top'), 0.25, 0.60),
            middle_bottom=parse_region(image_regions_data.get('middle_bottom'), 0.40, 0.75),
            bottom=parse_region(image_regions_data.get('bottom'), 0.75, 1.0)
        )
        
        # Parse person regions
        person_regions = PersonRegionsConfig(
            hat=parse_region(person_regions_data.get('hat'), 0.0, 0.15),
            glasses=parse_region(person_regions_data.get('glasses'), 0.1, 0.3),
            upper_stride=parse_region(person_regions_data.get('upper_stride'), 0.25, 0.65),
            upper_logo=parse_region(person_regions_data.get('upper_logo'), 0.25, 0.65),
            upper_plaid=parse_region(person_regions_data.get('upper_plaid'), 0.25, 0.65),
            upper_splice=parse_region(person_regions_data.get('upper_splice'), 0.25, 0.65),
            short_sleeve=parse_region(person_regions_data.get('short_sleeve'), 0.25, 0.5),
            long_sleeve=parse_region(person_regions_data.get('long_sleeve'), 0.25, 0.65),
            long_coat=parse_region(person_regions_data.get('long_coat'), 0.2, 0.8),
            lower_stripe=parse_region(person_regions_data.get('lower_stripe'), 0.5, 0.95),
            lower_pattern=parse_region(person_regions_data.get('lower_pattern'), 0.5, 0.95),
            trousers=parse_region(person_regions_data.get('trousers'), 0.5, 0.95),
            shorts=parse_region(person_regions_data.get('shorts'), 0.5, 0.8),
            skirt_dress=parse_region(person_regions_data.get('skirt_dress'), 0.3, 0.95),
            boots=parse_region(person_regions_data.get('boots'), 0.8, 1.0)
        )
        
        color_regions = ColorRegionsConfig(
            image=image_regions,
            person=person_regions
        )
        
        color_matching = ColorMatchingConfig(**color_filter_data.get('matching', {}))
        color_hsv = HSVConfig(**color_filter_data.get('hsv', {}))
        color_filter = ColorFilterConfig(
            regions=color_regions,
            matching=color_matching,
            hsv=color_hsv
        )
        
        clusterer_data = tasks_data.get('clusterer', {})
        clusterer_weights = ClustererWeightsConfig(**clusterer_data.get('weights', {}))
        clusterer = ClustererConfig(
            max_clusters=clusterer_data.get('max_clusters', 10),
            merge_threshold=clusterer_data.get('merge_threshold', 0.6),
            weights=clusterer_weights
        )
        
        tracker_data = tasks_data.get('tracker', {})
        tracker_confirmation = TrackerConfirmationConfig(**tracker_data.get('confirmation', {}))
        tracker_matching = TrackerMatchingConfig(**tracker_data.get('matching', {}))
        tracker_cropping = TrackerCroppingConfig(**tracker_data.get('cropping', {}))
        tracker_lifecycle = TrackerLifecycleConfig(**tracker_data.get('lifecycle', {}))
        tracker = TrackerConfig(
            confirmation=tracker_confirmation,
            matching=tracker_matching,
            cropping=tracker_cropping,
            lifecycle=tracker_lifecycle
        )
        
        tasks = TasksConfig(
            yolo=yolo,
            attributes=attributes,
            color_filter=color_filter,
            clusterer=clusterer,
            tracker=tracker,
        )

        # Parse optional smolvlm config
        smolvlm_data = tasks_data.get('smolvlm')
        if smolvlm_data is not None:
            tasks.smolvlm = SmolVLMConfig(**smolvlm_data)
        
        # Parse sinks
        sinks_data = data.get('sinks', {})
        mqtt = MQTTConfig(**sinks_data.get('mqtt', {}))
        sinks = SinksConfig(mqtt=mqtt)
        
        return cls(tasks=tasks, sinks=sinks)
    
    @classmethod
    def default(cls) -> 'MacOSDeviceConfig':
        """Create default configuration."""
        return cls(
            tasks=TasksConfig(
                yolo=YoloConfig(),
                attributes=AttributesConfig(),
                color_filter=ColorFilterConfig(
                    regions=ColorRegionsConfig(),
                    matching=ColorMatchingConfig(),
                    hsv=HSVConfig()
                ),
                clusterer=ClustererConfig(weights=ClustererWeightsConfig()),
                tracker=TrackerConfig(
                    confirmation=TrackerConfirmationConfig(),
                    matching=TrackerMatchingConfig(),
                    cropping=TrackerCroppingConfig(),
                    lifecycle=TrackerLifecycleConfig()
                )
            ),
            sinks=SinksConfig(mqtt=MQTTConfig())
        )
