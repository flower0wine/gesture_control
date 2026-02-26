"""
Hand Tracker Types
Defines data structures for hand tracking
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import IntEnum


class LandmarkIndex(IntEnum):
    """Hand landmark index reference"""

    WRIST = 0

    # Thumb
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4

    # Index finger
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8

    # Middle finger
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12

    # Ring finger
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16

    # Pinky
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


@dataclass
class Point:
    """2D/3D point with normalized coordinates"""

    x: float
    y: float
    z: float = 0.0

    def to_pixel(self, width: int, height: int) -> tuple:
        """Convert normalized coordinates to pixel coordinates"""
        return int(self.x * width), int(self.y * height)

    def to_pixel_3d(self, width: int, height: int) -> tuple:
        """Convert to pixel coordinates including z depth"""
        return int(self.x * width), int(self.y * height), self.z


@dataclass
class HandData:
    """Data for a single hand"""

    landmarks: List[Point]
    handedness: str  # "Left" or "Right"
    landmark_count: int = 21

    def get_landmark(self, index: LandmarkIndex) -> Point:
        """Get a specific landmark by index"""
        return self.landmarks[index]

    @property
    def index_finger_tip(self) -> Point:
        """Get index finger tip position"""
        return self.landmarks[LandmarkIndex.INDEX_FINGER_TIP]

    @property
    def wrist(self) -> Point:
        """Get wrist position"""
        return self.landmarks[LandmarkIndex.WRIST]

    @property
    def thumb_tip(self) -> Point:
        """Get thumb tip position"""
        return self.landmarks[LandmarkIndex.THUMB_TIP]


@dataclass
class TrackingResult:
    """Tracking result containing all detected hands"""

    hands: List[HandData]

    @property
    def hand_count(self) -> int:
        """Number of hands detected"""
        return len(self.hands)

    @property
    def left_hand(self) -> Optional[HandData]:
        """Get left hand if detected"""
        for hand in self.hands:
            if hand.handedness == "Left":
                return hand
        return None

    @property
    def right_hand(self) -> Optional[HandData]:
        """Get right hand if detected"""
        for hand in self.hands:
            if hand.handedness == "Right":
                return hand
        return None

    def get_hand_by_index(self, index: int) -> Optional[HandData]:
        """Get hand by index (0 or 1)"""
        if 0 <= index < len(self.hands):
            return self.hands[index]
        return None
