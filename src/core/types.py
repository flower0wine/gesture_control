"""
Core data types for hand tracking.

Defines all domain objects and enumerations.
"""

from dataclasses import dataclass
from enum import IntEnum
from typing import List, Tuple

import numpy as np


class Landmarks(IntEnum):
    """MediaPipe hand landmarks index."""

    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_FINGER_MCP = 5
    INDEX_FINGER_PIP = 6
    INDEX_FINGER_DIP = 7
    INDEX_FINGER_TIP = 8
    MIDDLE_FINGER_MCP = 9
    MIDDLE_FINGER_PIP = 10
    MIDDLE_FINGER_DIP = 11
    MIDDLE_FINGER_TIP = 12
    RING_FINGER_MCP = 13
    RING_FINGER_PIP = 14
    RING_FINGER_DIP = 15
    RING_FINGER_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20


# Indices for MCP joints (used in palm center calculation)
MCP_JOINT_INDICES: List[int] = [
    Landmarks.INDEX_FINGER_MCP,
    Landmarks.MIDDLE_FINGER_MCP,
    Landmarks.RING_FINGER_MCP,
    Landmarks.PINKY_MCP,
]


@dataclass
class HandLandmark:
    """Single hand landmark with 3D coordinates (normalized 0-1)."""

    x: float
    y: float
    z: float

    def screen_position(self, width: int, height: int) -> Tuple[int, int]:
        """
        Convert normalized coordinates to screen pixel position.

        Args:
            width: Screen width in pixels
            height: Screen height in pixels

        Returns:
            Tuple of (x, y) pixel coordinates
        """
        return (int(self.x * width), int(self.y * height))

    def distance_to(self, other: "HandLandmark") -> float:
        """Calculate Euclidean distance to another landmark."""
        return np.sqrt(
            (self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2
        )


@dataclass
class HandData:
    """Container for all hand detection results."""

    handedness: str  # "Left" or "Right"
    landmarks: List[HandLandmark]
    handedness_score: float

    @property
    def wrist(self) -> HandLandmark:
        """Get wrist position."""
        return self.landmarks[Landmarks.WRIST]

    @property
    def index_finger_tip(self) -> HandLandmark:
        """Get index finger tip position."""
        return self.landmarks[Landmarks.INDEX_FINGER_TIP]

    @property
    def middle_finger_tip(self) -> HandLandmark:
        """Get middle finger tip position."""
        return self.landmarks[Landmarks.MIDDLE_FINGER_TIP]

    @property
    def thumb_tip(self) -> HandLandmark:
        """Get thumb tip position."""
        return self.landmarks[Landmarks.THUMB_TIP]

    @property
    def pinky_tip(self) -> HandLandmark:
        """Get pinky tip position."""
        return self.landmarks[Landmarks.PINKY_TIP]

    @property
    def index_finger_mcp(self) -> HandLandmark:
        """Get index finger MCP (knuckle) position."""
        return self.landmarks[Landmarks.INDEX_FINGER_MCP]

    @property
    def middle_finger_mcp(self) -> HandLandmark:
        """Get middle finger MCP position."""
        return self.landmarks[Landmarks.MIDDLE_FINGER_MCP]

    @property
    def palm_center(self) -> HandLandmark:
        """Calculate approximate palm center as average of MCP joints."""
        center_x = sum(self.landmarks[i].x for i in MCP_JOINT_INDICES) / len(
            MCP_JOINT_INDICES
        )
        center_y = sum(self.landmarks[i].y for i in MCP_JOINT_INDICES) / len(
            MCP_JOINT_INDICES
        )
        center_z = sum(self.landmarks[i].z for i in MCP_JOINT_INDICES) / len(
            MCP_JOINT_INDICES
        )
        return HandLandmark(center_x, center_y, center_z)
