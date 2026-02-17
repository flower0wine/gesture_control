"""
Gesture module base types and enumerations.

Defines gesture types, results, and core interfaces.
"""

from dataclasses import dataclass
from enum import Enum, auto


class GestureType(Enum):
    """
    Enumeration of supported gesture types.

    Categories:
        - Static gestures: Hand poses (fist, peace, point, etc.)
        - Dynamic gestures: Movements (swipe, pinch, spread, etc.)
    """

    # Static gestures (hand poses)
    OPEN_HAND = auto()
    FIST = auto()
    PEACE = auto()
    POINT = auto()
    THUMBS_UP = auto()
    THUMBS_DOWN = auto()
    OK_SIGN = auto()
    ROCK_ON = auto()
    THREE_FINGERS = auto()

    # Dynamic gestures (movements)
    SWIPE_LEFT = auto()
    SWIPE_RIGHT = auto()
    SWIPE_UP = auto()
    SWIPE_DOWN = auto()
    PINCH = auto()
    SPREAD = auto()
    DRAG = auto()

    # System states
    UNKNOWN = auto()
    NO_HAND = auto()

    @property
    def is_static(self) -> bool:
        """Check if this is a static gesture (hand pose)."""
        static_gestures = {
            GestureType.OPEN_HAND,
            GestureType.FIST,
            GestureType.PEACE,
            GestureType.POINT,
            GestureType.THUMBS_UP,
            GestureType.THUMBS_DOWN,
            GestureType.OK_SIGN,
            GestureType.ROCK_ON,
            GestureType.THREE_FINGERS,
        }
        return self in static_gestures

    @property
    def is_dynamic(self) -> bool:
        """Check if this is a dynamic gesture (movement)."""
        dynamic_gestures = {
            GestureType.SWIPE_LEFT,
            GestureType.SWIPE_RIGHT,
            GestureType.SWIPE_UP,
            GestureType.SWIPE_DOWN,
            GestureType.PINCH,
            GestureType.SPREAD,
            GestureType.DRAG,
        }
        return self in dynamic_gestures

    @property
    def is_swipe(self) -> bool:
        """Check if this is a swipe gesture."""
        return self in {
            GestureType.SWIPE_LEFT,
            GestureType.SWIPE_RIGHT,
            GestureType.SWIPE_UP,
            GestureType.SWIPE_DOWN,
        }

    def to_display_name(self) -> str:
        """Convert gesture type to human-readable name."""
        names = {
            GestureType.OPEN_HAND: "Open Hand",
            GestureType.FIST: "Fist",
            GestureType.PEACE: "Peace",
            GestureType.POINT: "Pointing",
            GestureType.THUMBS_UP: "Thumbs Up",
            GestureType.THUMBS_DOWN: "Thumbs Down",
            GestureType.OK_SIGN: "OK Sign",
            GestureType.ROCK_ON: "Rock On",
            GestureType.THREE_FINGERS: "Three Fingers",
            GestureType.SWIPE_LEFT: "Swipe Left",
            GestureType.SWIPE_RIGHT: "Swipe Right",
            GestureType.SWIPE_UP: "Swipe Up",
            GestureType.SWIPE_DOWN: "Swipe Down",
            GestureType.PINCH: "Pinch",
            GestureType.SPREAD: "Spread",
            GestureType.DRAG: "Drag",
            GestureType.UNKNOWN: "Unknown",
            GestureType.NO_HAND: "No Hand",
        }
        return names.get(self, self.name)


@dataclass
class GestureResult:
    """
    Result of gesture recognition.

    Attributes:
        gesture: The recognized gesture type
        confidence: Confidence score (0.0 - 1.0)
        velocity: Movement velocity (for dynamic gestures)
        direction: Movement direction (for swipe gestures)
    """

    gesture: GestureType
    confidence: float
    velocity: float = 0.0
    direction: str | None = None

    def __post_init__(self) -> None:
        """Validate values after initialization."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.velocity = max(0.0, self.velocity)

    @property
    def is_valid(self) -> bool:
        """Check if the result is valid (hand detected and recognized)."""
        return self.gesture not in (GestureType.UNKNOWN, GestureType.NO_HAND)

    @property
    def is_high_confidence(self) -> bool:
        """Check if confidence is high (>= 0.7)."""
        return self.confidence >= 0.7

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "gesture": self.gesture.name,
            "gesture_display": self.gesture.to_display_name(),
            "confidence": round(self.confidence, 3),
            "velocity": round(self.velocity, 3),
            "direction": self.direction,
        }


# Gesture templates for static gesture matching
# Format: {gesture_name: {finger_name: is_extended}}
GESTURE_TEMPLATES: dict[str, dict[str, bool]] = {
    "open_hand": {
        "thumb": True,
        "index": True,
        "middle": True,
        "ring": True,
        "pinky": True,
    },
    "fist": {
        "thumb": False,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False,
    },
    "peace": {
        "thumb": False,
        "index": True,
        "middle": True,
        "ring": False,
        "pinky": False,
    },
    "point": {
        "thumb": False,
        "index": True,
        "middle": False,
        "ring": False,
        "pinky": False,
    },
    "thumbs_up": {
        "thumb": True,
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False,
    },
    "thumbs_down": {
        "thumb": True,  # Note: Need additional logic for down detection
        "index": False,
        "middle": False,
        "ring": False,
        "pinky": False,
    },
    "ok_sign": {
        "thumb": True,
        "index": True,  # Touching thumb
        "middle": True,
        "ring": True,
        "pinky": True,
    },
    "rock_on": {
        "thumb": False,
        "index": True,
        "middle": False,
        "ring": False,
        "pinky": True,
    },
    "three_fingers": {
        "thumb": False,
        "index": True,
        "middle": True,
        "ring": True,
        "pinky": False,
    },
}
