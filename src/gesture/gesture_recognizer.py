"""
Gesture recognizer.

Integrates finger state detection and swipe detection to provide
a unified gesture recognition interface.
"""

from typing import Optional

from src.core.types import HandData

from src.gesture.base import (
    GESTURE_TEMPLATES,
    GestureResult,
    GestureType,
)
from src.gesture.finger_detector import FingerStateDetector
from src.gesture.swipe_detector import SwipeDetector, SwipeDirection


class GestureRecognizer:
    """
    Unified gesture recognizer combining static pose and dynamic movement detection.

    This class integrates:
    - FingerStateDetector: Detects finger extension states (for static gestures)
    - SwipeDetector: Detects swipe movements (for dynamic gestures)

    Detection priority:
    1. Dynamic gestures (swipes) - checked first
    2. Static gestures (poses) - checked if no swipe

    Configuration:
    - enable_swipe: Enable swipe detection (default: True)
    - enable_static: Enable static gesture detection (default: True)
    - swipe_priority: If True, swipes take priority over static (default: True)
    """

    def __init__(
        self,
        enable_swipe: bool = True,
        enable_static: bool = True,
        swipe_priority: bool = True,
        extension_threshold: float = 1.0,
        swipe_min_distance: float = 0.02,
        swipe_smoothing: float = 0.3,
    ) -> None:
        """
        Initialize the gesture recognizer.

        Args:
            enable_swipe: Enable swipe detection
            enable_static: Enable static gesture detection
            swipe_priority: If True, swipe detection takes priority
            extension_threshold: Threshold for finger extension detection
            swipe_min_distance: Minimum distance for swipe detection
            swipe_smoothing: Velocity smoothing factor
        """
        self._enable_swipe = enable_swipe
        self._enable_static = enable_static
        self._swipe_priority = swipe_priority

        # Initialize detectors
        self._finger_detector = FingerStateDetector(
            extension_threshold=extension_threshold,
        )

        self._swipe_detector: Optional[SwipeDetector] = None
        if enable_swipe:
            self._swipe_detector = SwipeDetector(
                min_distance=swipe_min_distance,
                smoothing=swipe_smoothing,
            )

    @property
    def finger_detector(self) -> FingerStateDetector:
        """Get the finger state detector."""
        return self._finger_detector

    @property
    def swipe_detector(self) -> Optional[SwipeDetector]:
        """Get the swipe detector (if enabled)."""
        return self._swipe_detector

    def recognize(self, hand: HandData | None) -> GestureResult:
        """
        Recognize gesture from hand data.

        This is the main entry point for gesture recognition.
        Call this every frame with the current hand data.

        Args:
            hand: Hand data from tracker, or None if no hand detected

        Returns:
            GestureResult with gesture type, confidence, and metadata
        """
        # Handle no hand case
        if hand is None:
            return GestureResult(
                gesture=GestureType.NO_HAND,
                confidence=0.0,
            )

        # Priority-based detection
        if self._swipe_priority and self._enable_swipe:
            # Try swipe detection first
            swipe_result = self._detect_swipe(hand)
            if swipe_result is not None:
                return swipe_result

            # Fall back to static if no swipe
            if self._enable_static:
                return self._recognize_static(hand)

        else:
            # Try static first
            if self._enable_static:
                static_result = self._recognize_static(hand)
                if static_result.is_valid:
                    return static_result

            # Fall back to swipe
            if self._enable_swipe:
                swipe_result = self._detect_swipe(hand)
                if swipe_result is not None:
                    return swipe_result

        # No gesture detected
        return GestureResult(
            gesture=GestureType.UNKNOWN,
            confidence=0.0,
        )

    def recognize_static(self, hand: HandData) -> GestureResult:
        """
        Recognize only static gestures (hand poses).

        Args:
            hand: Hand data

        Returns:
            GestureResult for static gesture
        """
        return self._recognize_static(hand)

    def recognize_swipe(self, hand: HandData) -> GestureResult:
        """
        Recognize only swipe gestures.

        Args:
            hand: Hand data

        Returns:
            GestureResult for swipe gesture
        """
        if self._swipe_detector is None:
            return GestureResult(GestureType.UNKNOWN, 0.0)

        return self._detect_swipe(hand) or GestureResult(
            gesture=GestureType.UNKNOWN,
            confidence=0.0,
        )

    def _detect_swipe(self, hand: HandData) -> Optional[GestureResult]:
        """
        Detect swipe gesture.

        Args:
            hand: Hand data

        Returns:
            GestureResult if swipe detected, None otherwise
        """
        if self._swipe_detector is None:
            return None

        swipe_event = self._swipe_detector.update(hand)

        if swipe_event is None:
            return None

        # Map direction to gesture type
        direction_map = {
            SwipeDirection.LEFT: GestureType.SWIPE_LEFT,
            SwipeDirection.RIGHT: GestureType.SWIPE_RIGHT,
            SwipeDirection.UP: GestureType.SWIPE_UP,
            SwipeDirection.DOWN: GestureType.SWIPE_DOWN,
        }

        gesture_type = direction_map.get(swipe_event.direction, GestureType.UNKNOWN)

        # Calculate confidence based on velocity
        # Higher velocity = higher confidence (up to 1.0)
        confidence = min(swipe_event.velocity / 10.0, 1.0)

        return GestureResult(
            gesture=gesture_type,
            confidence=confidence,
            velocity=swipe_event.velocity,
            direction=swipe_event.direction_string,
        )

    def _recognize_static(self, hand: HandData) -> GestureResult:
        """
        Recognize static gesture (hand pose).

        Args:
            hand: Hand data

        Returns:
            GestureResult for static gesture
        """
        # Get finger states
        finger_states = self._finger_detector.get_finger_states_dict(hand)

        # Match against templates
        for gesture_name, template in GESTURE_TEMPLATES.items():
            if self._finger_detector.match_template(finger_states, template):
                # Map gesture name to enum
                enum_name = gesture_name.upper()
                # Handle special cases
                enum_name = enum_name.replace("_", "_")

                try:
                    gesture_type = GestureType[enum_name]
                except KeyError:
                    # Fallback for unmapped gestures
                    gesture_type = GestureType.UNKNOWN

                return GestureResult(
                    gesture=gesture_type,
                    confidence=0.9,  # High confidence for template match
                )

        # No match
        return GestureResult(
            gesture=GestureType.UNKNOWN,
            confidence=0.0,
        )

    def get_finger_states(self, hand: HandData) -> dict[str, bool]:
        """
        Get current finger extension states.

        Convenience method for debugging and visualization.

        Args:
            hand: Hand data

        Returns:
            Dictionary of finger states
        """
        return self._finger_detector.get_finger_states_dict(hand)

    def reset(self) -> None:
        """
        Reset detector state.

        Call this when switching users or contexts.
        """
        if self._swipe_detector is not None:
            self._swipe_detector.reset()

    def set_extension_threshold(self, threshold: float) -> None:
        """
        Set finger extension threshold.

        Args:
            threshold: New threshold value (0.5 - 1.5)
        """
        self._finger_detector.threshold = threshold

    def set_swipe_sensitivity(
        self, min_distance: float, smoothing: float = 0.3
    ) -> None:
        """
        Adjust swipe detection sensitivity.

        Args:
            min_distance: Minimum swipe distance (smaller = more sensitive)
            smoothing: Velocity smoothing (higher = smoother)
        """
        if self._swipe_detector is not None:
            self._swipe_detector.min_distance = min_distance
            # Note: Smoothing is set at init, would need to add setter


# Convenience factory functions


def create_swipe_only_recognizer(
    min_distance: float = 0.02,
    smoothing: float = 0.3,
) -> GestureRecognizer:
    """
    Create a recognizer that only detects swipes.

    Args:
        min_distance: Minimum swipe distance
        smoothing: Velocity smoothing

    Returns:
        Configured GestureRecognizer
    """
    return GestureRecognizer(
        enable_swipe=True,
        enable_static=False,
        swipe_priority=True,
        swipe_min_distance=min_distance,
        swipe_smoothing=smoothing,
    )


def create_static_only_recognizer(
    extension_threshold: float = 1.0,
) -> GestureRecognizer:
    """
    Create a recognizer that only detects static poses.

    Args:
        extension_threshold: Finger extension threshold

    Returns:
        Configured GestureRecognizer
    """
    return GestureRecognizer(
        enable_swipe=False,
        enable_static=True,
        extension_threshold=extension_threshold,
    )
