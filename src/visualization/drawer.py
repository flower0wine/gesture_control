"""
Visualization utilities for hand tracking.

Provides drawing functionality for hand landmarks and connections.
"""

import numpy as np
from mediapipe.tasks.python.components.containers.landmark import NormalizedLandmark
from mediapipe.tasks.python import vision

from src.core.types import HandData


class HandDrawer:
    """
    Draws hand landmarks on frames.

    Provides visualization functionality without coupling to tracking logic.
    """

    def __init__(self) -> None:
        """Initialize drawer with MediaPipe drawing utilities."""
        self._mp_drawing = vision.drawing_utils
        self._mp_drawing_styles = vision.drawing_styles
        self._mp_hands_connections = vision.HandLandmarksConnections

    def draw_landmarks(
        self,
        frame: np.ndarray,
        hand_data: HandData,
        draw_connections: bool = True,
    ) -> np.ndarray:
        """
        Draw hand landmarks on frame.

        Args:
            frame: Input frame (modified in place)
            hand_data: Hand data to draw
            draw_connections: Whether to draw connections between landmarks

        Returns:
            Frame with landmarks drawn
        """
        # Convert to MediaPipe format for drawing
        mp_landmarks = self._to_mp_landmarks(hand_data)

        if draw_connections:
            self._mp_drawing.draw_landmarks(
                frame,
                mp_landmarks,
                self._mp_hands_connections.HAND_CONNECTIONS,
                self._mp_drawing_styles.get_default_hand_landmarks_style(),
                self._mp_drawing_styles.get_default_hand_connections_style(),
            )
        else:
            self._mp_drawing.draw_landmarks(frame, mp_landmarks)

        return frame

    def draw_landmarks_batch(
        self,
        frame: np.ndarray,
        hands_data: list[HandData],
        draw_connections: bool = True,
    ) -> np.ndarray:
        """
        Draw multiple hands on frame.

        Args:
            frame: Input frame
            hands_data: List of hand data to draw
            draw_connections: Whether to draw connections

        Returns:
            Frame with all hands drawn
        """
        for hand_data in hands_data:
            self.draw_landmarks(frame, hand_data, draw_connections)
        return frame

    @staticmethod
    def _to_mp_landmarks(hand_data: HandData) -> list[NormalizedLandmark]:
        """
        Convert HandData to MediaPipe landmark format.

        Args:
            hand_data: Hand data to convert

        Returns:
            List of MediaPipe NormalizedLandmark objects
        """
        mp_landmarks = []
        for lm in hand_data.landmarks:
            landmark = NormalizedLandmark()
            landmark.x = lm.x
            landmark.y = lm.y
            landmark.z = lm.z
            mp_landmarks.append(landmark)
        return mp_landmarks
