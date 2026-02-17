"""
Hand tracking module using MediaPipe.

Provides hand detection and landmark extraction without coupling to visualization.
"""

from typing import List

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from src.core.config import TrackerConfig
from src.core.types import HandData, HandLandmark
from src.tracking.model_manager import ModelManager


class HandTracker:
    """
    Hand tracker using MediaPipe.

    Detects hands and extracts 21 3D landmarks per hand.
    Returns raw detection data without visualization coupling.
    """

    def __init__(self, config: TrackerConfig | None = None) -> None:
        """
        Initialize the hand tracker.

        Args:
            config: Tracker configuration. Uses defaults if not provided.
        """
        self._config = config or TrackerConfig()
        self._model_manager = ModelManager()
        self._init_landmarker()

    def _init_landmarker(self) -> None:
        """Initialize MediaPipe hand landmarker."""
        model_path = self._model_manager.model_path_str
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=self._config.max_hands,
            min_hand_detection_confidence=self._config.min_detection_confidence,
            min_hand_presence_confidence=self._config.min_tracking_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )
        self._hand_landmarker = vision.HandLandmarker.create_from_options(options)

    def process_frame(
        self, frame: np.ndarray, timestamp: int = 0, flipped: bool = False
    ) -> List[HandData]:
        """
        Process a video frame and detect hands.

        Args:
            frame: BGR image frame from OpenCV
            timestamp: Frame timestamp in milliseconds
            flipped: Whether the frame is horizontally flipped (selfie view).
                     If True, left/right handedness will be swapped.

        Returns:
            List of HandData objects, one per detected hand
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._hand_landmarker.detect_for_video(mp_image, timestamp)
        return self._convert_results(results, flipped)

    def process_image(self, frame: np.ndarray) -> List[HandData]:
        """
        Process a static image and detect hands.

        Args:
            frame: BGR image frame from OpenCV

        Returns:
            List of HandData objects, one per detected hand
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._hand_landmarker.detect(mp_image)
        return self._convert_results(results, flipped=False)

    def _convert_results(
        self, results: vision.HandLandmarkerResult, flipped: bool
    ) -> List[HandData]:
        """
        Convert MediaPipe results to HandData objects.

        Args:
            results: Raw MediaPipe detection results
            flipped: Whether to swap handedness (for selfie view)

        Returns:
            List of HandData objects
        """
        if not results.hand_landmarks or not results.handedness:
            return []

        hands_data: List[HandData] = []
        for landmarks, handedness in zip(results.hand_landmarks, results.handedness):
            landmark_list = [HandLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in landmarks]

            hand_label = handedness[0].category_name
            hand_score = handedness[0].score

            # Swap handedness if frame is flipped (selfie view)
            if flipped:
                hand_label = "Left" if hand_label == "Right" else "Right"

            hands_data.append(
                HandData(
                    handedness=hand_label,
                    landmarks=landmark_list,
                    handedness_score=hand_score,
                )
            )

        return hands_data

    def close(self) -> None:
        """Release resources."""
        self._hand_landmarker.close()

    @property
    def config(self) -> TrackerConfig:
        """Get tracker configuration."""
        return self._config
