"""
Configuration settings for AutoATC backend
"""

from pydantic_settings import BaseSettings
from typing import List, Optional
import os
from pathlib import Path

class Settings(BaseSettings):
    """Application settings."""
    
    # Application settings
    APP_NAME: str = "AutoATC API"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database settings
    DATABASE_URL: str = "sqlite:///./atc.db"
    DATABASE_ECHO: bool = False
    
    # CORS settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501"
    ]
    
    # File upload settings
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    UPLOAD_DIR: str = "static/uploads"
    
    # AI Model settings
    DETECTION_MODEL_PATH: str = "models/yolov8n.pt"
    BREED_MODEL_PATH: str = "models/breed_classifier.pt"
    DISEASE_MODEL_PATH: str = "models/disease_detector.pt"
    KEYPOINT_MODEL_COMPLEXITY: int = 1
    ARUCO_DICTIONARY: int = 6
    ARUCO_MARKER_SIZE: float = 5.0
    
    # Confidence thresholds
    DETECTION_CONFIDENCE: float = 0.5
    KEYPOINT_CONFIDENCE: float = 0.5
    BREED_CONFIDENCE: float = 0.6
    DISEASE_CONFIDENCE: float = 0.5
    
    # BPA Integration settings
    BPA_API_URL: Optional[str] = None
    BPA_API_KEY: Optional[str] = None
    BPA_TIMEOUT: int = 30
    
    # Redis settings (for caching)
    REDIS_URL: Optional[str] = None
    CACHE_TTL: int = 3600  # 1 hour
    
    # Logging settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # Security settings
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Processing settings
    MAX_PROCESSING_TIME: int = 300  # 5 minutes
    ENABLE_BACKGROUND_TASKS: bool = True
    
    # Storage settings
    STATIC_DIR: str = "static"
    IMAGES_DIR: str = "static/images"
    ANNOTATED_DIR: str = "static/annotated"
    MODELS_DIR: str = "models"
    
    # Validation settings
    ENABLE_VALIDATION: bool = True
    VALIDATION_THRESHOLD: float = 0.7
    
    # Export settings
    ENABLE_BPA_EXPORT: bool = True
    EXPORT_RETRY_LIMIT: int = 3
    EXPORT_BATCH_SIZE: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True

def get_settings() -> Settings:
    """Get application settings."""
    return Settings()

# Create necessary directories
def create_directories():
    """Create necessary directories."""
    directories = [
        Settings().STATIC_DIR,
        Settings().IMAGES_DIR,
        Settings().ANNOTATED_DIR,
        Settings().MODELS_DIR,
        Settings().UPLOAD_DIR
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

# Initialize settings
settings = get_settings()
create_directories()