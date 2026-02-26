"""
Mouse Controller
Control mouse cursor using hand landmarks
"""

import time
import numpy as np
import pyautogui
from typing import Optional, Tuple

from .types import HandData, Point, LandmarkIndex


class MouseController:
    """
    Mouse controller using hand tracking.
    Maps index finger tip position to mouse cursor position.
    """

    def __init__(
        self,
        screen_width: Optional[int] = None,
        screen_height: Optional[int] = None,
        smoothing: float = 0.3,
        offset_x: int = 0,
        offset_y: int = 0,
        enabled: bool = True,
    ):
        """
        Initialize mouse controller.

        Args:
            screen_width: Screen width (auto-detected if None)
            screen_height: Screen height (auto-detected if None)
            smoothing: Smoothing factor (0.0-1.0). Lower = smoother but slower
            offset_x: X offset for mouse position
            offset_y: Y offset for mouse position
            enabled: Whether mouse control is enabled
        """
        # Get screen size
        if screen_width is None or screen_height is None:
            screen_width, screen_height = pyautogui.size()

        self.screen_width = screen_width
        self.screen_height = screen_height
        self.smoothing = smoothing
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.enabled = enabled

        # Previous position for smoothing
        self._prev_x = screen_width // 2
        self._prev_y = screen_height // 2

        # Click state
        self._is_clicking = False
        self._click_threshold = 0.05  # Distance threshold for click detection

        # Rate limiting
        self._last_update_time = time.time()
        self._min_update_interval = 0.01  # ~100 FPS max

    def update(self, hand_data: HandData) -> None:
        """
        Update mouse position based on hand data.

        Args:
            hand_data: Hand data containing landmarks
        """
        if not self.enabled or hand_data is None:
            return

        # Rate limiting
        current_time = time.time()
        if current_time - self._last_update_time < self._min_update_interval:
            return
        self._last_update_time = current_time

        # Get index finger tip position
        finger_tip = hand_data.index_finger_tip

        # Convert normalized coordinates to screen coordinates
        target_x = int(finger_tip.x * self.screen_width)
        target_y = int(finger_tip.y * self.screen_height)

        # Apply offset
        target_x += self.offset_x
        target_y += self.offset_y

        # Clamp to screen bounds
        target_x = max(0, min(self.screen_width - 1, target_x))
        target_y = max(0, min(self.screen_height - 1, target_y))

        # Apply smoothing
        if self.smoothing > 0:
            self._prev_x = int(
                self._prev_x * (1 - self.smoothing) + target_x * self.smoothing
            )
            self._prev_y = int(
                self._prev_y * (1 - self.smoothing) + target_y * self.smoothing
            )
            target_x = self._prev_x
            target_y = self._prev_y

        # Move mouse
        pyautogui.moveTo(target_x, target_y)

    def is_clicking(self, hand_data: HandData) -> bool:
        """
        Check if hand is in clicking position.
        Clicking is detected when thumb tip is close to index finger tip.

        Args:
            hand_data: Hand data containing landmarks

        Returns:
            True if clicking gesture detected
        """
        if hand_data is None:
            return False

        thumb_tip = hand_data.thumb_tip
        index_tip = hand_data.index_finger_tip

        # Calculate distance between thumb and index tip
        distance = np.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
        )

        return distance < self._click_threshold

    def click(self) -> None:
        """Perform a mouse click"""
        pyautogui.click()

    def right_click(self) -> None:
        """Perform a right mouse click"""
        pyautogui.rightClick()

    def double_click(self) -> None:
        """Perform a double click"""
        pyautogui.doubleClick()

    def scroll(self, direction: str, amount: int = 3) -> None:
        """
        Scroll the mouse wheel.

        Args:
            direction: "up" or "down"
            amount: Number of scroll units
        """
        if direction.lower() == "up":
            pyautogui.scroll(amount)
        else:
            pyautogui.scroll(-amount)

    def enable(self) -> None:
        """Enable mouse control"""
        self.enabled = True

    def disable(self) -> None:
        """Disable mouse control"""
        self.enabled = False

    def toggle(self) -> bool:
        """Toggle mouse control and return new state"""
        self.enabled = not self.enabled
        return self.enabled

    def set_smoothing(self, value: float) -> None:
        """Set smoothing factor (0.0-1.0)"""
        self.smoothing = max(0.0, min(1.0, value))

    def set_offset(self, x: int, y: int) -> None:
        """Set offset for mouse position"""
        self.offset_x = x
        self.offset_y = y

    @property
    def is_enabled(self) -> bool:
        """Check if mouse control is enabled"""
        return self.enabled

    @property
    def position(self) -> Tuple[int, int]:
        """Get current mouse position"""
        return pyautogui.position()
