"""
Gesture recognition module.

Provides comprehensive gesture detection including:
- Static gesture detection (hand poses)
- Dynamic gesture detection (swipes, movements)

Main components:
- GestureRecognizer: Unified interface for all gesture detection
- FingerStateDetector: Detects finger extension states
- SwipeDetector: Detects swipe gestures

Usage:
    from src.gesture import GestureRecognizer

    recognizer = GestureRecognizer()
    result = recognizer.recognize(hand_data)
    print(result.gesture, result.confidence)
"""

from src.gesture.base import (
    GESTURE_TEMPLATES,
    GestureResult,
    GestureType,
)
from src.gesture.finger_detector import (
    FingerState,
    FingerStateDetector,
    HandState,
)
from src.gesture.gesture_recognizer import (
    create_static_only_recognizer,
    create_swipe_only_recognizer,
    GestureRecognizer,
)
from src.gesture.swipe_detector import (
    SwipeDetector,
    SwipeDirection,
    SwipeEvent,
)

__all__ = [
    # Base types
    "GestureType",
    "GestureResult",
    "GESTURE_TEMPLATES",
    # Finger detection
    "FingerStateDetector",
    "FingerState",
    "HandState",
    # Swipe detection
    "SwipeDetector",
    "SwipeDirection",
    "SwipeEvent",
    # Main recognizer
    "GestureRecognizer",
    # Factory functions
    "create_swipe_only_recognizer",
    "create_static_only_recognizer",
]
