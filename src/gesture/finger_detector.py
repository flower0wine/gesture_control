"""
Finger state detector.

Detects whether each finger is extended or curled based on landmark positions.
"""

from dataclasses import dataclass
from typing import Literal

from src.core.types import HandData, Landmarks


# Finger names
FingerName = Literal["thumb", "index", "middle", "ring", "pinky"]


# Configuration for each finger: (tip_index, pip_index, mcp_index)
# Used to determine finger extension state
FINGER_CONFIG: dict[FingerName, tuple[int, int, int]] = {
    "thumb": (Landmarks.THUMB_TIP, Landmarks.THUMB_IP, Landmarks.THUMB_CMC),
    "index": (
        Landmarks.INDEX_FINGER_TIP,
        Landmarks.INDEX_FINGER_PIP,
        Landmarks.INDEX_FINGER_MCP,
    ),
    "middle": (
        Landmarks.MIDDLE_FINGER_TIP,
        Landmarks.MIDDLE_FINGER_PIP,
        Landmarks.MIDDLE_FINGER_MCP,
    ),
    "ring": (
        Landmarks.RING_FINGER_TIP,
        Landmarks.RING_FINGER_PIP,
        Landmarks.RING_FINGER_MCP,
    ),
    "pinky": (Landmarks.PINKY_TIP, Landmarks.PINKY_PIP, Landmarks.PINKY_MCP),
}


@dataclass
class FingerState:
    """
    State of a single finger.

    Attributes:
        name: Finger name (thumb, index, middle, ring, pinky)
        is_extended: Whether the finger is extended (True) or curled (False)
        extension_ratio: Ratio of tip distance to pip distance (>1 means extended)
    """

    name: FingerName
    is_extended: bool
    extension_ratio: float


@dataclass
class HandState:
    """
    Complete state of a hand (all fingers).

    Attributes:
        fingers: Dictionary of finger states
        handedness: Handedness label ("Left" or "Right")
        is_valid: Whether the hand data is valid
    """

    fingers: dict[FingerName, FingerState]
    handedness: str
    is_valid: bool = True

    @property
    def extended_fingers(self) -> list[FingerName]:
        """Get list of extended fingers."""
        return [name for name, state in self.fingers.items() if state.is_extended]

    @property
    def curled_fingers(self) -> list[FingerName]:
        """Get list of curled fingers."""
        return [name for name, state in self.fingers.items() if not state.is_extended]

    @property
    def num_extended(self) -> int:
        """Get count of extended fingers."""
        return sum(1 for state in self.fingers.values() if state.is_extended)


class FingerStateDetector:
    """
    Detects the state (extended/curled) of each finger.

    The detection is based on comparing distances from each finger's
    landmarks to the wrist:
    - If tip is farther from wrist than pip joint -> finger is extended
    - If tip is closer to wrist than pip joint -> finger is curled

    The extension ratio (tip_distance / pip_distance) determines the threshold:
    - ratio > extension_threshold -> extended
    - ratio <= extension_threshold -> curled

    Typical threshold values:
    - 0.85-0.95: More sensitive (fewer false positives)
    - 0.95-1.05: Balanced
    - 1.05-1.15: More conservative (fewer false negatives)
    """

    # All finger names in order (thumb to pinky)
    ALL_FINGERS: tuple[FingerName, ...] = ("thumb", "index", "middle", "ring", "pinky")

    def __init__(
        self,
        extension_threshold: float = 1.0,
        use_wrist_reference: bool = True,
    ) -> None:
        """
        Initialize the finger state detector.

        Args:
            extension_threshold: Threshold ratio for determining extension.
                                Values > 1.0 mean tip must be noticeably farther
                                from wrist than pip joint.
            use_wrist_reference: If True, use wrist as reference point.
                                 If False, use palm center (experimental).
        """
        if not 0.5 <= extension_threshold <= 1.5:
            msg = f"extension_threshold must be between 0.5 and 1.5, got {extension_threshold}"
            raise ValueError(msg)

        self._threshold = extension_threshold
        self._use_wrist_reference = use_wrist_reference

    @property
    def threshold(self) -> float:
        """Get the current extension threshold."""
        return self._threshold

    @threshold.setter
    def threshold(self, value: float) -> None:
        """Set the extension threshold."""
        if not 0.5 <= value <= 1.5:
            msg = f"extension_threshold must be between 0.5 and 1.5, got {value}"
            raise ValueError(msg)
        self._threshold = value

    def is_extended(self, hand: HandData, finger_name: FingerName) -> bool:
        """
        Determine if a specific finger is extended.

        Args:
            hand: Hand data containing landmarks
            finger_name: Name of the finger to check

        Returns:
            True if finger is extended, False if curled
        """
        tip_idx, pip_idx, _ = FINGER_CONFIG[finger_name]

        tip = hand.landmarks[tip_idx]
        pip = hand.landmarks[pip_idx]

        if self._use_wrist_reference:
            wrist = hand.wrist
            tip_distance = tip.distance_to(wrist)
            pip_distance = pip.distance_to(wrist)
        else:
            # Use palm center as reference
            palm = hand.palm_center
            tip_distance = tip.distance_to(palm)
            pip_distance = pip.distance_to(palm)

        # Avoid division by zero
        if pip_distance == 0:
            return tip_distance > 0

        ratio = tip_distance / pip_distance
        return ratio > self._threshold

    def get_extension_ratio(self, hand: HandData, finger_name: FingerName) -> float:
        """
        Get the extension ratio for a finger.

        The ratio is: (tip to reference distance) / (pip to reference distance)
        - ratio > 1.0: tip is farther than pip (finger likely extended)
        - ratio < 1.0: tip is closer than pip (finger likely curled)
        - ratio ≈ 1.0: tip and pip are at similar distance (uncertain)

        Args:
            hand: Hand data containing landmarks
            finger_name: Name of the finger

        Returns:
            Extension ratio (tip_distance / pip_distance)
        """
        tip_idx, pip_idx, _ = FINGER_CONFIG[finger_name]

        tip = hand.landmarks[tip_idx]
        pip = hand.landmarks[pip_idx]

        if self._use_wrist_reference:
            wrist = hand.wrist
            tip_distance = tip.distance_to(wrist)
            pip_distance = pip.distance_to(wrist)
        else:
            palm = hand.palm_center
            tip_distance = tip.distance_to(palm)
            pip_distance = pip.distance_to(palm)

        if pip_distance == 0:
            return 0.0

        return tip_distance / pip_distance

    def get_finger_state(self, hand: HandData, finger_name: FingerName) -> FingerState:
        """
        Get detailed state for a single finger.

        Args:
            hand: Hand data containing landmarks
            finger_name: Name of the finger

        Returns:
            FingerState with name, extension status, and ratio
        """
        ratio = self.get_extension_ratio(hand, finger_name)
        is_extended = ratio > self._threshold

        return FingerState(
            name=finger_name,
            is_extended=is_extended,
            extension_ratio=ratio,
        )

    def detect_hand_state(self, hand: HandData) -> HandState:
        """
        Detect the state of all fingers on a hand.

        Args:
            hand: Hand data containing landmarks

        Returns:
            HandState with all finger states
        """
        fingers: dict[FingerName, FingerState] = {}

        for finger_name in self.ALL_FINGERS:
            fingers[finger_name] = self.get_finger_state(hand, finger_name)

        return HandState(
            fingers=fingers,
            handedness=hand.handedness,
            is_valid=True,
        )

    def get_finger_states_dict(self, hand: HandData) -> dict[FingerName, bool]:
        """
        Get finger states as a simple dictionary.

        This is a convenience method returning a simple dict
        suitable for gesture template matching.

        Args:
            hand: Hand data containing landmarks

        Returns:
            Dictionary mapping finger name to extension state
            Example: {"thumb": True, "index": False, ...}
        """
        return {
            finger_name: self.is_extended(hand, finger_name)
            for finger_name in self.ALL_FINGERS
        }

    def match_template(
        self,
        finger_states: dict[FingerName, bool],
        template: dict[FingerName, bool],
    ) -> bool:
        """
        Match finger states against a template.

        Args:
            finger_states: Current finger states
            template: Expected finger states

        Returns:
            True if all template conditions match
        """
        return all(
            finger_states.get(finger) == expected
            for finger, expected in template.items()
        )
