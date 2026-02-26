"""
Hand Detector
Core hand landmark detection implementation
"""

import cv2
import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .types import HandData, Point, TrackingResult, LandmarkIndex


class HandDetector:
    """
    Hand landmark detector.
    Provides 21 keypoints per hand without gesture recognition.
    """

    def __init__(
        self,
        num_hands: int = 2,
        model_path: str = "hand_landmarker.task",
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        """
        Initialize the hand detector.

        Args:
            num_hands: Maximum number of hands to detect (1-2)
            model_path: Path to the hand landmarker model file
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self._num_hands = num_hands
        self._model_path = model_path
        self._min_detection_confidence = min_detection_confidence
        self._min_tracking_confidence = min_tracking_confidence

        # Create the hand landmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            running_mode=vision.RunningMode.VIDEO,
        )

        self._landmarker = vision.HandLandmarker.create_from_options(options)

        # Import drawing utilities
        self._mp_drawing = vision.drawing_utils
        self._mp_drawing_styles = vision.drawing_styles

    def detect(self, frame_rgb, frame_idx: int) -> TrackingResult:
        """
        Detect hand landmarks from a frame.

        Args:
            frame_rgb: RGB image (OpenCV format)
            frame_idx: Frame index for video mode

        Returns:
            TrackingResult containing detected hands
        """
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self._landmarker.detect_for_video(mp_image, frame_idx)

        hands = []

        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                # Get handedness
                handedness = "Unknown"
                if result.handedness and len(result.handedness) > i:
                    raw_handedness = result.handedness[i][0].category_name
                    # Fix: In selfie mode (flipped), swap Left/Right
                    handedness = "Right" if raw_handedness == "Left" else "Left"

                # Convert landmarks to Point objects
                landmarks = [
                    Point(
                        x=lm.x,
                        y=lm.y,
                        z=lm.z,
                    )
                    for lm in hand_landmarks
                ]

                hands.append(
                    HandData(
                        landmarks=landmarks,
                        handedness=handedness,
                    )
                )

        return TrackingResult(hands=hands)

    def detect_from_frame(self, frame, frame_idx: int) -> TrackingResult:
        """
        Detect hand landmarks from a BGR frame (auto-converts to RGB).

        Args:
            frame: BGR image (OpenCV format)
            frame_idx: Frame index for video mode

        Returns:
            TrackingResult containing detected hands
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return self.detect(frame_rgb, frame_idx)

    def draw_landmarks(
        self,
        image,
        result: TrackingResult,
        show_labels: bool = False,
    ) -> None:
        """
        Draw hand landmarks on image.

        Args:
            image: Image to draw on (modified in place)
            result: Tracking result
            show_labels: Whether to show landmark indices
        """
        if not result.hands:
            return

        h, w, _ = image.shape
        colors = [(0, 255, 0), (255, 0, 255)]

        for hand_idx, hand_data in enumerate(result.hands):
            color = colors[hand_idx % len(colors)]

            # Convert our Point objects back to MediaPipe format for drawing
            # (or draw manually - here's a simple approach)
            hand_landmarks = hand_data.landmarks

            # Draw connections manually for better control
            self._draw_hand_connections(image, hand_landmarks, color, w, h)

            # Draw landmarks
            for lm in hand_landmarks:
                px, py = int(lm.x * w), int(lm.y * h)
                cv2.circle(image, (px, py), 4, color, -1)

            # Draw hand label
            wrist = hand_data.wrist
            label_x = int(wrist.x * w)
            label_y = int(wrist.y * h) - 20

            label = f"{hand_data.handedness} Hand"
            cv2.putText(
                image,
                label,
                (label_x - 40, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )

            # Optionally show landmark numbers
            if show_labels:
                for lm_idx, lm in enumerate(hand_landmarks):
                    if lm_idx % 4 == 0:
                        px, py = int(lm.x * w), int(lm.y * h)
                        cv2.putText(
                            image,
                            str(lm_idx),
                            (px + 5, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

    def _draw_hand_connections(self, image, landmarks, color, w, h):
        """Draw hand connections (bones)"""
        # Define connections between landmarks
        connections = [
            # Wrist to thumb
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
            # Wrist to index
            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),
            # Wrist to middle
            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),
            # Wrist to ring
            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),
            # Wrist to pinky
            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),
            # Palm connections
            (5, 9),
            (9, 13),
            (13, 17),
        ]

        for start_idx, end_idx in connections:
            start = landmarks[start_idx]
            end = landmarks[end_idx]

            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)

            cv2.line(image, (x1, y1), (x2, y2), color, 2)

    def __del__(self):
        """Cleanup"""
        if hasattr(self, "_landmarker"):
            self._landmarker.close()
