"""
Swipe detector.

Detects swipe gestures based on fingertip movement tracking.
Provides smooth velocity calculation and direction detection.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import numpy as np

from src.core.types import HandData, HandLandmark


class SwipeDirection(Enum):
    """Swipe direction enumeration."""

    LEFT = auto()
    RIGHT = auto()
    UP = auto()
    DOWN = auto()

    def to_string(self) -> str:
        """Convert to lowercase string."""
        return self.name.lower()


# All valid swipe directions
SWIPE_DIRECTIONS = (
    SwipeDirection.LEFT,
    SwipeDirection.RIGHT,
    SwipeDirection.UP,
    SwipeDirection.DOWN,
)


@dataclass
class SwipeEvent:
    """
    Represents a detected swipe event.

    Attributes:
        direction: Swipe direction
        velocity: Movement velocity (normalized distance per frame)
        distance: Total swipe distance
        start_position: Starting position (normalized coordinates)
        current_position: Current position (normalized coordinates)
        delta: Movement delta (dx, dy)
        is_tracking: Whether currently tracking this swipe
        finger: Which finger is being tracked
    """

    direction: SwipeDirection
    velocity: float
    distance: float
    start_x: float
    start_y: float
    current_x: float
    current_y: float
    delta_x: float
    delta_y: float
    is_tracking: bool
    finger: str = "index"

    @property
    def direction_string(self) -> str:
        """Get direction as string."""
        return self.direction.to_string()

    @property
    def displacement(self) -> tuple[float, float]:
        """Get displacement from start to current."""
        return (self.current_x - self.start_x, self.current_y - self.start_y)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "direction": self.direction_string,
            "velocity": round(self.velocity, 4),
            "distance": round(self.distance, 4),
            "start": (round(self.start_x, 4), round(self.start_y, 4)),
            "current": (round(self.current_x, 4), round(self.current_y, 4)),
            "delta": (round(self.delta_x, 4), round(self.delta_y, 4)),
        }


class SwipeDetector:
    """
    Detects swipe gestures by tracking fingertip movement.

    The detector:
    1. Tracks a specific fingertip (default: index finger)
    2. Calculates movement delta between frames
    3. Applies smoothing to velocity
    4. Determines swipe direction based on dominant axis
    5. Emits swipe events when threshold is crossed

    Configuration:
    - min_distance: Minimum movement to consider (normalized 0-1)
    - velocity_factor: Multiplier for velocity calculation
    - smoothing: Low-pass filter coefficient (0-1)
    - direction_threshold: Minimum ratio for dominant axis
    """

    def __init__(
        self,
        min_distance: float = 0.02,
        velocity_factor: float = 50.0,
        smoothing: float = 0.3,
        direction_threshold: float = 1.5,
        track_finger: str = "index",
    ) -> None:
        """
        Initialize the swipe detector.

        Args:
            min_distance: Minimum movement to trigger swipe detection
                         (normalized 0-1, default 0.02)
            velocity_factor: Multiplier for velocity calculation
            smoothing: Smoothing factor for velocity (0-1, lower=smoother)
            direction_threshold: Minimum ratio for dominant axis
                                (e.g., 1.5 means dx must be 1.5x dy)
            track_finger: Which finger to track ("index", "middle", etc.)
        """
        # Validate parameters
        if not 0.001 <= min_distance <= 0.5:
            msg = f"min_distance must be between 0.001 and 0.5, got {min_distance}"
            raise ValueError(msg)
        if not 0.0 <= smoothing <= 1.0:
            msg = f"smoothing must be between 0.0 and 1.0, got {smoothing}"
            raise ValueError(msg)
        if not 1.0 <= direction_threshold <= 10.0:
            msg = f"direction_threshold must be between 1.0 and 10.0, got {direction_threshold}"
            raise ValueError(msg)

        self._min_distance = min_distance
        self._velocity_factor = velocity_factor
        self._smoothing = smoothing
        self._direction_threshold = direction_threshold
        self._track_finger = track_finger

        # Internal state
        self._prev_hand: Optional[HandData] = None
        self._prev_position: Optional[tuple[float, float]] = None
        self._prev_velocity: float = 0.0
        self._current_swipe: Optional[SwipeEvent] = None
        self._swipe_start_position: Optional[tuple[float, float]] = None
        self._is_swiping: bool = False

    @property
    def min_distance(self) -> float:
        """Get minimum distance threshold."""
        return self._min_distance

    @min_distance.setter
    def min_distance(self, value: float) -> None:
        """Set minimum distance threshold."""
        if not 0.001 <= value <= 0.5:
            msg = f"min_distance must be between 0.001 and 0.5, got {value}"
            raise ValueError(msg)
        self._min_distance = value

    @property
    def is_tracking(self) -> bool:
        """Check if currently tracking a swipe."""
        return self._is_swiping

    def _get_fingertip(self, hand: HandData) -> HandLandmark:
        """Get the fingertip to track based on configuration."""
        from src.core.types import Landmarks

        finger_map = {
            "thumb": Landmarks.THUMB_TIP,
            "index": Landmarks.INDEX_FINGER_TIP,
            "middle": Landmarks.MIDDLE_FINGER_TIP,
            "ring": Landmarks.RING_FINGER_TIP,
            "pinky": Landmarks.PINKY_TIP,
        }

        landmark_idx = finger_map.get(self._track_finger, Landmarks.INDEX_FINGER_TIP)
        return hand.landmarks[landmark_idx]

    def _determine_direction(self, dx: float, dy: float) -> Optional[SwipeDirection]:
        """
        Determine swipe direction based on movement deltas.

        Uses direction_threshold to ensure movement is predominantly
        along one axis.
        """
        abs_dx = abs(dx)
        abs_dy = abs(dy)

        # Check if movement exceeds threshold
        distance = np.sqrt(dx**2 + dy**2)
        if distance < self._min_distance:
            return None

        # Check dominant axis
        if abs_dx >= abs_dy * self._direction_threshold:
            # Horizontal movement dominates
            return SwipeDirection.RIGHT if dx > 0 else SwipeDirection.LEFT
        elif abs_dy >= abs_dx * self._direction_threshold:
            # Vertical movement dominates
            return SwipeDirection.DOWN if dy > 0 else SwipeDirection.UP

        return None

    def _calculate_velocity(self, raw_velocity: float) -> float:
        """Apply smoothing to velocity calculation."""
        if self._smoothing > 0:
            # Low-pass filter: smoothed = alpha * raw + (1 - alpha) * prev
            smoothed = (
                self._smoothing * raw_velocity
                + (1 - self._smoothing) * self._prev_velocity
            )
            self._prev_velocity = smoothed
            return smoothed
        return raw_velocity

    def update(self, hand: HandData | None) -> Optional[SwipeEvent]:
        """
        Update swipe detection with new hand data.

        Call this method every frame with the current hand data.
        Returns a SwipeEvent if a swipe is detected, None otherwise.

        Args:
            hand: Current frame's hand data, or None if no hand detected

        Returns:
            SwipeEvent if swipe detected, None otherwise
        """
        # Handle no hand detected
        if hand is None:
            return self._handle_no_hand()

        # Get current fingertip position
        current_tip = self._get_fingertip(hand)
        current_position = (current_tip.x, current_tip.y)

        # First frame - just initialize
        if self._prev_hand is None:
            self._prev_position = current_position
            self._prev_hand = hand
            self._current_swipe = None
            return None

        # Calculate movement delta
        prev_x, prev_y = self._prev_position
        dx = current_tip.x - prev_x
        dy = current_tip.y - prev_y
        distance = np.sqrt(dx**2 + dy**2)

        # Check if this is a swipe
        direction = self._determine_direction(dx, dy)

        if direction is None:
            # Movement too small or diagonal - not a swipe
            if self._is_swiping:
                # End current swipe
                self._is_swiping = False
                self._swipe_start_position = None
            self._current_swipe = None
        else:
            # Valid swipe detected
            raw_velocity = distance * self._velocity_factor
            velocity = self._calculate_velocity(raw_velocity)

            # Start new swipe or update existing
            if not self._is_swiping:
                self._is_swiping = True
                self._swipe_start_position = prev_x, prev_y

            start_x, start_y = self._swipe_start_position or (prev_x, prev_y)

            self._current_swipe = SwipeEvent(
                direction=direction,
                velocity=velocity,
                distance=distance,
                start_x=start_x,
                start_y=start_y,
                current_x=current_tip.x,
                current_y=current_tip.y,
                delta_x=dx,
                delta_y=dy,
                is_tracking=self._is_swiping,
                finger=self._track_finger,
            )

        # Update previous state
        self._prev_position = current_position
        self._prev_hand = hand

        return self._current_swipe

    def _handle_no_hand(self) -> Optional[SwipeEvent]:
        """Handle case when no hand is detected."""
        # Return final swipe event before losing tracking
        final_event = self._current_swipe

        # Reset internal state
        self._prev_hand = None
        self._prev_position = None
        self._prev_velocity = 0.0
        self._current_swipe = None
        self._is_swiping = False
        self._swipe_start_position = None

        return final_event

    def reset(self) -> None:
        """
        Reset the detector state.

        Call this to clear all tracking state, for example
        when switching to a different tracking mode.
        """
        self._prev_hand = None
        self._prev_position = None
        self._prev_velocity = 0.0
        self._current_swipe = None
        self._is_swiping = False
        self._swipe_start_position = None

    def get_position_delta(self) -> tuple[float, float]:
        """
        Get the raw position delta without swipe interpretation.

        Returns:
            (dx, dy) movement vector since last frame
        """
        if self._prev_position is None:
            return (0.0, 0.0)

        # This would need to be tracked - simplified for now
        return (0.0, 0.0)
