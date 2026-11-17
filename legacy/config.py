"""Configuration settings for the VoiceStudio project.

All paths are relative to the repository root. Settings can be overridden
via a `.env` file placed in the project root.
"""
from pathlib import Path
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings

class Settings(BaseSettings):
    # Core directories
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    PLUGINS_ROOT: Path = PROJECT_ROOT / "model_plugins"
    MODEL_STORAGE: Path = PROJECT_ROOT / "models"
    AUDIO_STORAGE: Path = PROJECT_ROOT / "static" / "generated"

    # Runtime options
    QUANTIZE: bool = False  # Enable 8‑bit quantization for supported models
    DEFAULT_DEVICE: str = "auto"  # cpu / cuda / rocm / auto

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "{time} | {level} | {message}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance used throughout the codebase
settings = Settings()
