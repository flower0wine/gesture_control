"""
Video capture utilities.

Provides camera capture functionality with context manager support.
"""

from typing import Tuple

import cv2
import numpy as np

from src.core.config import CameraConfig


class CameraError(Exception):
    """Base exception for camera-related errors."""

    pass


class CameraOpenError(CameraError):
    """Raised when camera cannot be opened."""

    pass


class VideoCapture:
    """
    Wrapper for OpenCV video capture.

    Provides context manager support and configurable settings.
    """

    _cap: cv2.VideoCapture

    def __init__(self, config: CameraConfig | None = None) -> None:
        """
        Initialize video capture.

        Args:
            config: Camera configuration. Uses defaults if not provided.

        Raises:
            CameraOpenError: If camera cannot be opened
        """
        self._config = config or CameraConfig()
        self._cap = cv2.VideoCapture(self._config.index)

        if not self._cap.isOpened():
            msg = f"Cannot open camera {self._config.index}"
            raise CameraOpenError(msg)

        # Configure frame size
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._config.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._config.height)

    def read(self) -> Tuple[bool, np.ndarray | None]:
        """
        Read next frame.

        Returns:
            Tuple of (success, frame)
        """
        return self._cap.read()

    @property
    def frame_size(self) -> Tuple[int, int]:
        """Get actual frame size (width, height)."""
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    @property
    def fps(self) -> float:
        """Get camera FPS."""
        return self._cap.get(cv2.CAP_PROP_FPS)

    def release(self) -> None:
        """Release the camera."""
        self._cap.release()

    def __enter__(self) -> "VideoCapture":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.release()

    def __del__(self) -> None:
        """Ensure camera is released."""
        try:
            self.release()
        except Exception:
            pass
