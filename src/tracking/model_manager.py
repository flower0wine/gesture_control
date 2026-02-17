"""
Model manager for downloading and managing MediaPipe models.

Handles model download, caching, and path resolution.
"""

import urllib.request
from pathlib import Path

from src.core.config import MODEL_FILENAME, MODEL_URL


class ModelError(Exception):
    """Base exception for model-related errors."""

    pass


class ModelDownloadError(ModelError):
    """Raised when model download fails."""

    pass


class ModelNotFoundError(ModelError):
    """Raised when model file is not found."""

    pass


class ModelManager:
    """
    Manages MediaPipe hand landmarker model.

    Handles download, caching, and path resolution.
    """

    def __init__(self, models_dir: Path | None = None) -> None:
        """
        Initialize model manager.

        Args:
            models_dir: Directory to store models. Defaults to project/models.
        """
        if models_dir is None:
            models_dir = Path(__file__).parent.parent.parent / "models"
        self._models_dir = Path(models_dir)
        self._model_filename = MODEL_FILENAME
        self._model_url = MODEL_URL

    @property
    def model_path(self) -> Path:
        """Get the path to the model file, downloading if necessary."""
        model_path = self._models_dir / self._model_filename
        if not model_path.exists():
            self._download_model()
        return model_path

    @property
    def model_path_str(self) -> str:
        """Get model path as string for MediaPipe."""
        return str(self.model_path)

    def _download_model(self) -> None:
        """Download the model from remote URL."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        model_path = self._models_dir / self._model_filename

        print("Downloading MediaPipe hand landmarker model...")
        print(f"URL: {self._model_url}")

        try:
            urllib.request.urlretrieve(self._model_url, model_path)
            print(f"Model downloaded to {model_path}")
        except Exception as e:
            msg = f"Failed to download model: {e}"
            raise ModelDownloadError(msg) from e

    def ensure_model_exists(self) -> bool:
        """
        Check if model exists, download if missing.

        Returns:
            True if model is available, False otherwise.
        """
        return self.model_path.exists()

    def get_model_info(self) -> dict:
        """Get information about the model file."""
        model_path = self.model_path
        return {
            "path": str(model_path),
            "size_bytes": model_path.stat().st_size if model_path.exists() else 0,
            "url": self._model_url,
        }
