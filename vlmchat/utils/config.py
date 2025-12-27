"""
Application configuration using Pydantic for validation and type safety.
"""

import logging
import json
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field, validator, ConfigDict
from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum

# Assuming platform_detect is in a sibling 'utils' directory
from .platform_detect import Platform, detect_platform


# --- Enums ---

class HistoryFormat(str, Enum):
    """Supported conversation history formats."""
    XML = "xml"
    MINIMAL = "minimal"
    NONE = "none"


class LogLevel(str, Enum):
    """Supported logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class CameraModel(str, Enum):
    """Supported camera hardware models."""
    IMX500 = "imx500"
    IMX477 = "imx477"
    IMX219 = "imx219"
    IMAGE_LIBRARY = "image_library"
    NONE = "none"


class Device(str, Enum):
    """Camera device identifiers."""
    CAMERA0 = "camera0"
    CAMERA1 = "camera1"


# --- Configuration Models ---

class ModelConfig(BaseModel):
    """Model configuration parameters."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    model_path: str = Field(
        default="HuggingFaceTB/SmolVLM2-256M-Instruct",
        description="Path to the SmolVLM model on HuggingFace Hub or local path",
        min_length=1
    )
    max_new_tokens: int = Field(
        default=1024,
        ge=1,
        le=4096,
        description="Maximum number of tokens to generate in a response"
    )
    eos_token_id: int = Field(
        default=198,
        ge=0,
        description="Token ID representing end-of-sequence (typically newline)"
    )
    use_onnx: bool = Field(
        default=True,
        description="Whether to use ONNX runtime for faster inference"
    )
    onnx_base_path: Path = Field(
        default=Path("~/onnx").expanduser().absolute(),
        description="Base directory for ONNX model files"
    )
    
    # CLIP Model Configuration
    clip_model_name: str = Field(
        default="MobileCLIP2-S0",
        description="Name of the CLIP model to use"
    )
    clip_pretrained_path: Optional[str] = Field(
        default="./mobileclip2_s0.pt",
        description="Path to the pretrained CLIP model file"
    )
    clip_model_kwargs: Optional[dict] = Field(
        default=None,
        description="Additional keyword arguments for CLIP model initialization"
    )
    
    # FashionClip Model Configuration
    fashion_clip_model_name: str = Field(
        default="hf-hub:Marqo/marqo-fashionSigLIP",
        description="Name of the FashionClip model to use"
    )
    fashion_clip_pretrained: str = Field(
        default="",
        description="Pretrained weights for FashionClip (empty for default)"
    )
    fashion_clip_model_kwargs: Optional[dict] = Field(
        default=None,
        description="Additional keyword arguments for FashionClip model initialization"
    )
    device: str = Field(
        default="cpu",
        description="Device to run models on (cpu, cuda, mps)"
    )
    
    # TensorRT Engine Paths
    yolo_engine_path: Path = Field(
        default=Path("~/Dev/model-rt-build/platform/jetson/release_artifacts/yolov8n_fp16.engine").expanduser().absolute(),
        description="Path to YOLO TensorRT engine file"
    )
    yolo_model_path: Path = Field(
        default=Path("~/yolov8n.pt").expanduser().absolute(),
        description="Path to YOLO Ultralytics model file (.pt)"
    )
    clip_image_engine_path: Path = Field(
        default=Path("~/Dev/model-rt-build/platform/jetson/release_artifacts/image_fp16.engine").expanduser().absolute(),
        description="Path to CLIP image encoder TensorRT engine file"
    )
    clip_text_engine_path: Path = Field(
        default=Path("~/Dev/model-rt-build/platform/jetson/release_artifacts/text_fp16.engine").expanduser().absolute(),
        description="Path to CLIP text encoder TensorRT engine file"
    )

    @validator('onnx_base_path', pre=True)
    def expand_onnx_path(cls, v):
        """Expand user path and validate ONNX base path."""
        return Path(v).expanduser().absolute()
    
    @validator('yolo_engine_path', 'clip_image_engine_path', 'clip_text_engine_path', pre=True)
    def expand_engine_paths(cls, v):
        """Expand user paths for TensorRT engine files."""
        return Path(v).expanduser().absolute()

    def get_onnx_model_path(self) -> Path:
        """Get the full ONNX model path."""
        model_name = Path(self.model_path).name
        return self.onnx_base_path.joinpath(model_name)


class ConversationConfig(BaseModel):
    """Conversation history configuration parameters."""
    model_config = ConfigDict(extra='forbid')

    max_pairs: int = Field(
        default=10,
        ge=1,
        le=1000,
        description="Maximum number of conversation pairs to retain"
    )
    max_images: int = Field(
        default=1,
        ge=1,
        le=10,
        description="Maximum number of images to keep in context"
    )
    history_format: HistoryFormat = Field(
        default=HistoryFormat.XML,
        description="Format for conversation history (XML or MINIMAL)"
    )
    word_limit: int = Field(
        default=15,
        ge=1,
        le=100,
        description="Word limit for minimal formatter text truncation"
    )


class LoggingConfig(BaseModel):
    """Logging configuration parameters."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Default logging level"
    )
    format: str = Field(
        default='%(asctime)s - %(levelname)s - %(message)s',
        description="Log format string",
        min_length=1
    )

    @validator('format')
    def validate_log_format(cls, v):
        """Validate log format string."""
        try:
            test_record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="test", args=(), exc_info=None
            )
            fmt = logging.Formatter(v)
            fmt.format(test_record)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid log format string: {e}")
        return v


class PathsConfig(BaseModel):
    """File paths and directories configuration."""
    model_config = ConfigDict(extra='forbid', str_strip_whitespace=True)

    project_root: Path = Field(
        default=Path(__file__).parent.absolute(),
        description="Application root directory (auto-detected)"
    )
    coco_labels_path: str = Field(
        default="assets/coco_labels.txt",
        description="Path to COCO labels file relative to project root",
    )
    captured_images_dir: Path = Field(
        default=Path("~/captured_images").expanduser().absolute(),
        description="Directory for saving captured images",
    )
    metrics_file: Path = Field(
        default=Path("metrics.json"),
        description="Path to load VLM metrics JSON file"
    )
    pipeline_dirs: list[str] = Field(
        default=["~/pipelines", "./pipelines"],
        description="Directories to search for pipeline DSL files"
    )

    @validator('project_root', pre=True, always=True)
    def set_project_root(cls, v):
        """Set project root if not provided."""
        if v is None:
            return Path(__file__).parent.absolute()
        return Path(v)


class CameraConfig(BaseModel):
    """Camera configuration parameters."""
    model_config = ConfigDict(extra='forbid')

    camera_model: CameraModel = Field(
        default=CameraModel.NONE,
        description="Camera model to use (imx500, imx477, imx219, image_library, none)"
    )
    camera_device: Device = Field(
        default=Device.CAMERA0,
        description="Camera device identifier (camera0, camera1)"
    )
    width: int = Field(
        default=640,
        description="Camera image width. Must be a mode the camera supports."
    )
    height: int = Field(
        default=480,
        description="Camera image height. Must be a mode the camera supports."
    )
    framerate: int = Field(
        default=5,
        description="Camera frame rate in FPS."
    )
    image_library_dir: Path = Field(
        default=Path("~/test-images").expanduser().absolute(),
        description="Base directory for ONNX model files"
    )


class ClustererConfig(BaseModel):
    """Object clusterer configuration parameters."""
    model_config = ConfigDict(extra='forbid')

    max_clusters: int = Field(
        default=4,
        ge=1,
        le=100,
        description="Maximum number of clusters to create"
    )
    merge_threshold: float = Field(
        default=1.5,
        gt=0.0,
        description="Stop merging if best cost is above this threshold"
    )
    proximity_weight: float = Field(
        default=1.2,
        ge=0.0,
        description="Weight for spatial packing efficiency cost"
    )
    size_weight: float = Field(
        default=1.2,
        ge=0.0,
        description="Weight for size difference cost"
    )
    semantic_pair_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for semantic pair similarity cost"
    )
    semantic_single_weight: float = Field(
        default=1.0,
        ge=0.0,
        description="Weight for semantic single vs prompts cost"
    )
    user_prompts: list[str] = Field(
        default=[
            "a group of people riding horses",
            "people on horseback",
            "riders on a trail"
        ],
        description="User prompts for semantic clustering"
    )


class VLMChatConfig(BaseSettings):
    """
Main application configuration.

Loads settings from environment variables automatically.
Example:
VLMCHAT_MODEL__MODEL_PATH=/path/to/model
VLMCHAT_LOGGING__LEVEL=DEBUG
VLMCHAT_PLATFORM=rpi
"""
    model_config = SettingsConfigDict(
        env_prefix="VLMCHAT_",
        env_nested_delimiter="__",
        extra='forbid'
    )

    model: ModelConfig = Field(default_factory=ModelConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    camera: CameraConfig = Field(default_factory=CameraConfig)
    clusterer: ClustererConfig = Field(default_factory=ClustererConfig)

    platform: Platform = Field(
        default_factory=detect_platform,
        description="Runtime platform (rpi, jetson, etc.). Can be overridden."
    )

    @classmethod
    def load_from_file(cls, config_path: str) -> 'VLMChatConfig':
        """
        Load configuration from a JSON or YAML file.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                if config_file.suffix.lower() in ['.yaml', '.yml']:
                    try:
                        import yaml
                        data = yaml.safe_load(f)
                    except ImportError:
                        raise ValueError("PyYAML is required to load YAML configuration files")
                else:
                    data = json.load(f)

            return cls(**data)
        except Exception as e:
            raise ValueError(f"Failed to load configuration from {config_path}: {e}")

    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to a JSON file.
        """
        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        data = self.model_dump_json(indent=2)
        config_file.write_text(data, encoding='utf-8')


def create_default_config_file(config_path: str = "config.json") -> None:
    """
    Create a default configuration file with current settings.
    """
    default_config = VLMChatConfig()
    default_config.save_to_file(config_path)
    print(f"Default configuration saved to: {config_path}")


if __name__ == "__main__":
    # Create a sample configuration file when run directly
    create_default_config_file()