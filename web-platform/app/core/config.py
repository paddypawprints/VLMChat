"""
Application configuration and settings.
"""
import os
from functools import lru_cache
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Database
    database_url: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")
    
    # Session
    session_secret_key: str = "dev-secret-key-change-in-production"
    session_expire_hours: int = 24
    
    # File upload
    max_upload_size: int = 10 * 1024 * 1024  # 10MB
    allowed_image_types: list[str] = ["image/jpeg", "image/png", "image/gif", "image/webp"]
    
    # Device communication
    device_connect_timeout: float = 1.0
    device_scan_timeout: float = 2.0
    
    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    """Get cached application settings."""
    return Settings()