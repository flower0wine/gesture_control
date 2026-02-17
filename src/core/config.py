"""
Configuration classes for hand tracking.

Provides centralized configuration management with sensible defaults.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Final

# Model configuration
MODEL_URL: Final[str] = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_FILENAME: Final[str] = "hand_landmarker.task"


@dataclass
class TrackerConfig:
    """Configuration for HandTracker."""

    max_hands: int = 1
    """Maximum number of hands to detect."""

    model_complexity: int = 0
    """Model complexity (0=lightest, 1=medium, 2=heavy)."""

    min_detection_confidence: float = 0.5
    """Minimum hand detection confidence [0, 1]."""

    min_tracking_confidence: float = 0.5
    """Minimum hand tracking confidence [0, 1]."""

    @classmethod
    def from_defaults(cls) -> "TrackerConfig":
        """Create config with default values."""
        return cls()


@dataclass
class CameraConfig:
    """Configuration for video capture."""

    index: int = 0
    """Camera device index."""

    width: int = 640
    """Frame width in pixels."""

    height: int = 480
    """Frame height in pixels."""

    @classmethod
    def from_defaults(cls) -> "CameraConfig":
        """Create config with default values."""
        return cls()


@dataclass
class AppConfig:
    """Application-level configuration."""

    tracker: TrackerConfig
    camera: CameraConfig
    models_dir: Path

    @classmethod
    def from_defaults(cls, models_dir: Path | None = None) -> "AppConfig":
        """Create config with default values."""
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / "models"
        return cls(
            tracker=TrackerConfig.from_defaults(),
            camera=CameraConfig.from_defaults(),
            models_dir=models_dir,
        )
