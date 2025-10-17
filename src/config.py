"""
Application configuration using Pydantic for validation and type safety.

This module defines the configuration schema for the VLMChat application,
including model settings, conversation history parameters, logging configuration,
and file paths. Configuration can be loaded from environment variables or
configuration files.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, validator, ConfigDict
from enum import Enum
from pathlib import Path as _Path
from typing import Optional as _Optional
from utils.camera_base import Platform as _Platform


class HistoryFormat(str, Enum):
    """Supported conversation history formats."""
    XML = "xml"
    MINIMAL = "minimal"


class LogLevel(str, Enum):
    """Supported logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelConfig(BaseModel):
    """Model configuration parameters."""
    model_config = ConfigDict(extra='forbid')

    model_path: str = Field(
        default="HuggingFaceTB/SmolVLM2-256M-Instruct",
        description="Path to the SmolVLM model on HuggingFace Hub or local path"
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
    onnx_base_path: str = Field(
        default="~/onnx",
        description="Base directory for ONNX model files"
    )

    @validator('model_path')
    def validate_model_path(cls, v):
        """Validate model path format."""
        if not v.strip():
            raise ValueError("Model path cannot be empty")
        return v.strip()

    @validator('onnx_base_path')
    def expand_onnx_path(cls, v):
        """Expand user path and validate ONNX base path."""
        from pathlib import Path
        expanded_path = Path(v).expanduser().absolute()
        return str(expanded_path)

    def get_onnx_model_path(self) -> Path:
        """
        Get the full ONNX model path by combining base path with model path.

        Returns:
            Path: Full path to ONNX model directory
        """
        from pathlib import Path
        import hashlib

        # Create a safe directory name from model path
        model_name = self.model_path.split('/')[-1] if '/' in self.model_path else self.model_path

        return Path(expand_path(self.onnx_base_path)) / model_name


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
    model_config = ConfigDict(extra='forbid')

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Default logging level"
    )
    format: str = Field(
        default='%(asctime)s - %(levelname)s - %(message)s',
        description="Log format string"
    )

    @validator('format')
    def validate_log_format(cls, v):
        """Validate log format string."""
        if not v.strip():
            raise ValueError("Log format cannot be empty")
        # Test format string validity
        try:
            # Use logging.Formatter to validate the format. Formatter will
            # populate fields like asctime when formatting a record, so this
            # correctly accepts common logging format strings such as
            # '%(asctime)s - %(levelname)s - %(message)s'.
            test_record = logging.LogRecord(
                name="test", level=logging.INFO, pathname="", lineno=0,
                msg="test", args=(), exc_info=None
            )
            fmt = logging.Formatter(v)
            # Formatter.format may mutate the record but that's fine for a
            # validation check; any exception indicates an invalid format.
            fmt.format(test_record)
        except (KeyError, ValueError, TypeError) as e:
            raise ValueError(f"Invalid log format string: {e}")
        return v.strip()


class PathsConfig(BaseModel):
    """File paths and directories configuration."""
    model_config = ConfigDict(extra='forbid')

    project_root: Optional[Path] = Field(
        default=None,
        description="Application root directory (auto-detected if not specified)"
    )
    coco_labels_path: str = Field(
        default="assets/coco_labels.txt",
        description="Path to COCO labels file relative to project root"
    )
    captured_images_dir: str = Field(
        default="captured_images",
        description="Directory for saving captured images"
    )

    @validator('project_root', pre=True)
    def set_project_root(cls, v):
        """Set project root if not provided."""
        if v is None:
            # Auto-detect project root from this file's location
            return Path(__file__).parent.absolute()
        return Path(v) if isinstance(v, str) else v

    @validator('coco_labels_path', 'captured_images_dir')
    def validate_paths(cls, v):
        """Validate path strings."""
        if not v.strip():
            raise ValueError("Path cannot be empty")
        return v.strip()


class VLMChatConfig(BaseModel):
    """Main application configuration."""
    model_config = ConfigDict(extra='forbid')

    model: ModelConfig = Field(default_factory=ModelConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    # runtime platform detected at startup (not loaded from file)
    runtime_platform: Optional[str] = Field(
        default=None,
        description="Detected runtime platform (rpi, jetson)"
    )

    def get_runtime_platform(self) -> _Optional[_Platform]:
        """Return the runtime platform as a `Platform` enum if set/recognizable.

        This helper converts the stored string value into the `Platform` enum
        and accepts common synonyms (e.g., 'raspberry', 'rpi', 'jetson').
        Returns None when no valid platform is available.
        """
        val = self.runtime_platform
        if not val:
            return None
        val = str(val).strip().lower()
        if val in ("rpi", "raspberry", "raspberrypi"):
            return _Platform.RPI
        if val in ("jetson", "nvidia"):
            return _Platform.JETSON
        # try direct enum lookup
        try:
            return _Platform(val)
        except Exception:
            return None

    @classmethod
    def load_from_file(cls, config_path: str) -> 'VLMChatConfig':
        """
        Load configuration from a JSON or YAML file.

        Args:
            config_path: Path to configuration file

        Returns:
            VLMChatConfig: Loaded and validated configuration

        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config file is invalid
        """
        import json
        from pathlib import Path

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

    @classmethod
    def load_from_env(cls, prefix: str = "VLMCHAT_") -> 'VLMChatConfig':
        """
        Load configuration from environment variables.

        Args:
            prefix: Environment variable prefix (default: VLMCHAT_)

        Returns:
            VLMChatConfig: Configuration with values from environment variables
        """
        import os

        config_dict = {}

        # Model configuration
        model_dict = {}
        if f"{prefix}MODEL_PATH" in os.environ:
            model_dict["model_path"] = os.environ[f"{prefix}MODEL_PATH"]
        if f"{prefix}MAX_NEW_TOKENS" in os.environ:
            model_dict["max_new_tokens"] = int(os.environ[f"{prefix}MAX_NEW_TOKENS"])
        if f"{prefix}EOS_TOKEN_ID" in os.environ:
            model_dict["eos_token_id"] = int(os.environ[f"{prefix}EOS_TOKEN_ID"])
        if f"{prefix}USE_ONNX" in os.environ:
            model_dict["use_onnx"] = os.environ[f"{prefix}USE_ONNX"].lower() in ('true', '1', 'yes')
        if f"{prefix}ONNX_BASE_PATH" in os.environ:
            model_dict["onnx_base_path"] = os.environ[f"{prefix}ONNX_BASE_PATH"]
        if model_dict:
            config_dict["model"] = model_dict

        # Conversation configuration
        conversation_dict = {}
        if f"{prefix}MAX_PAIRS" in os.environ:
            conversation_dict["max_pairs"] = int(os.environ[f"{prefix}MAX_PAIRS"])
        if f"{prefix}MAX_IMAGES" in os.environ:
            conversation_dict["max_images"] = int(os.environ[f"{prefix}MAX_IMAGES"])
        if f"{prefix}HISTORY_FORMAT" in os.environ:
            conversation_dict["history_format"] = os.environ[f"{prefix}HISTORY_FORMAT"]
        if f"{prefix}WORD_LIMIT" in os.environ:
            conversation_dict["word_limit"] = int(os.environ[f"{prefix}WORD_LIMIT"])
        if conversation_dict:
            config_dict["conversation"] = conversation_dict

        # Logging configuration
        logging_dict = {}
        if f"{prefix}LOG_LEVEL" in os.environ:
            logging_dict["level"] = os.environ[f"{prefix}LOG_LEVEL"]
        if f"{prefix}LOG_FORMAT" in os.environ:
            logging_dict["format"] = os.environ[f"{prefix}LOG_FORMAT"]
        if logging_dict:
            config_dict["logging"] = logging_dict

        # Paths configuration
        paths_dict = {}
        if f"{prefix}PROJECT_ROOT" in os.environ:
            paths_dict["project_root"] = os.environ[f"{prefix}PROJECT_ROOT"]
        if f"{prefix}COCO_LABELS_PATH" in os.environ:
            paths_dict["coco_labels_path"] = os.environ[f"{prefix}COCO_LABELS_PATH"]
        if f"{prefix}CAPTURED_IMAGES_DIR" in os.environ:
            paths_dict["captured_images_dir"] = os.environ[f"{prefix}CAPTURED_IMAGES_DIR"]
        if paths_dict:
            config_dict["paths"] = paths_dict

        # Allow runtime platform override from environment
        if f"{prefix}RUNTIME_PLATFORM" in os.environ:
            if "runtime_platform" not in config_dict:
                config_dict.setdefault("runtime_platform", os.environ[f"{prefix}RUNTIME_PLATFORM"]) 

        return cls(**config_dict)

    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to a JSON file.

        Args:
            config_path: Path where to save the configuration
        """
        import json
        from pathlib import Path

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Convert to dict and handle Path objects
        data = self.model_dump()
        if 'paths' in data and 'project_root' in data['paths']:
            data['paths']['project_root'] = str(data['paths']['project_root'])

        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# Global configuration instance
_config: Optional[VLMChatConfig] = None


def get_config() -> VLMChatConfig:
    """
    Get the global configuration instance.

    Returns:
        VLMChatConfig: Global configuration instance
    """
    global _config
    if _config is None:
        _config = VLMChatConfig()
    return _config


def set_config(config: VLMChatConfig) -> None:
    """
    Set the global configuration instance.

    Args:
        config: New configuration to use globally
    """
    global _config
    _config = config


def load_config(config_path: Optional[str] = None) -> VLMChatConfig:
    """
    Load configuration from file or environment, set as global config.

    Args:
        config_path: Optional path to configuration file

    Returns:
        VLMChatConfig: Loaded configuration
    """
    if config_path:
        config = VLMChatConfig.load_from_file(config_path)
    else:
        # Try to load from environment, fall back to defaults
        config = VLMChatConfig.load_from_env()

    set_config(config)
    return config


def create_default_config_file(config_path: str = "config.json") -> None:
    """
    Create a default configuration file with current settings.

    Args:
        config_path: Path where to create the configuration file
    """
    default_config = VLMChatConfig()
    default_config.save_to_file(config_path)
    print(f"Default configuration saved to: {config_path}")

def expand_path(file_path: str) -> str:
    """
    Expands a file path to an absolute path.

    If the path is relative, it prepends the user's home directory.
    If the path is already absolute, it returns it unchanged.

    Args:
        file_path: The file path to expand.

    Returns:
        The absolute file path.
    """
    # os.path.expanduser handles paths starting with '~'
    expanded_path = os.path.expanduser(file_path)

    # If the path is still not absolute, join it with the home directory
    if not os.path.isabs(expanded_path):
        home_dir = os.path.expanduser('~')
        return os.path.join(home_dir, expanded_path)

    return expanded_path

if __name__ == "__main__":
    # Create a sample configuration file when run directly
    create_default_config_file()

