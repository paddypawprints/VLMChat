"""Camera Framework configuration (shared/generic pipeline settings)."""

from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import yaml


@dataclass
class PipelineConfig:
    """Pipeline-level configuration."""
    max_workers: int = 4
    memory_leak_threshold: float = 30.0
    stats_interval: int = 30


@dataclass
class CameraSourceConfig:
    """Physical camera source configuration."""
    width: int = 1920
    height: int = 1080
    fps: float = 30.0
    device: int = 0


@dataclass
class ImageLibrarySourceConfig:
    """Image library source configuration."""
    image_dir: str = "/Users/patrick/Dev/MOT15"
    width: int = 1920
    height: int = 1080
    framerate: float = 1.0


@dataclass
class SourcesConfig:
    """Sources configuration."""
    camera: CameraSourceConfig
    image_library: ImageLibrarySourceConfig


@dataclass
class BuffersConfig:
    """Buffer configuration."""
    default_size: int = 5
    alert_size: int = 10
    policy: str = "drop_oldest"


@dataclass
class CameraFrameworkConfig:
    """Top-level camera framework configuration."""
    pipeline: PipelineConfig
    sources: SourcesConfig
    buffers: BuffersConfig
    
    @classmethod
    def load(cls, path: str) -> 'CameraFrameworkConfig':
        """Load configuration from YAML file.
        
        Args:
            path: Path to YAML config file
            
        Returns:
            CameraFrameworkConfig instance
            
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
        
        # Parse nested structure
        pipeline = PipelineConfig(**data.get('pipeline', {}))
        
        sources_data = data.get('sources', {})
        camera = CameraSourceConfig(**sources_data.get('camera', {}))
        image_library = ImageLibrarySourceConfig(**sources_data.get('image_library', {}))
        sources = SourcesConfig(camera=camera, image_library=image_library)
        
        buffers = BuffersConfig(**data.get('buffers', {}))
        
        return cls(
            pipeline=pipeline,
            sources=sources,
            buffers=buffers
        )
    
    @classmethod
    def default(cls) -> 'CameraFrameworkConfig':
        """Create default configuration."""
        return cls(
            pipeline=PipelineConfig(),
            sources=SourcesConfig(
                camera=CameraSourceConfig(),
                image_library=ImageLibrarySourceConfig()
            ),
            buffers=BuffersConfig()
        )
