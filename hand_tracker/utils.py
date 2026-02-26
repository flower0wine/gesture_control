"""
Utilities
Common utility functions for hand tracking
"""

import cv2
import time
from typing import Tuple


class FPSCounter:
    """Calculate FPS (frames per second)"""

    def __init__(self, update_interval: int = 10):
        """
        Args:
            update_interval: Update FPS display every N frames
        """
        self._update_interval = update_interval
        self._frame_count = 0
        self._start_time = time.time()
        self._fps = 0.0

    def update(self) -> float:
        """
        Update FPS counter.

        Returns:
            Current FPS value
        """
        self._frame_count += 1

        if self._frame_count >= self._update_interval:
            elapsed = time.time() - self._start_time
            self._fps = self._frame_count / elapsed
            self._frame_count = 0
            self._start_time = time.time()

        return self._fps

    @property
    def fps(self) -> float:
        """Get current FPS"""
        return self._fps


def draw_info_panel(
    image,
    fps: float,
    width: int,
    height: int,
    hand_count: int,
    mouse_enabled: bool = True,
    show_labels: bool = False,
) -> None:
    """
    Draw info panel on image.

    Args:
        image: Image to draw on
        fps: Current FPS
        width: Frame width
        height: Frame height
        hand_count: Number of hands detected
        mouse_enabled: Whether mouse control is enabled
        show_labels: Whether landmark labels are shown
    """
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 240, 150

    # Semi-transparent background
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    # Border
    cv2.rectangle(
        image,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (100, 100, 100),
        1,
    )

    # Info lines
    fps_color = (0, 255, 0) if fps > 20 else (0, 0, 255)
    mouse_color = (0, 255, 0) if mouse_enabled else (100, 100, 100)

    info_lines = [
        ("Hand Tracking + Mouse Control", (0, 255, 255)),
        ("", (255, 255, 255)),
        (f"FPS: {fps:.1f}", fps_color),
        (f"Resolution: {width}x{height}", (255, 255, 255)),
        (f"Hands: {hand_count}", (255, 255, 0)),
        ("", (255, 255, 255)),
        (f"Mouse: {'ON' if mouse_enabled else 'OFF'}", mouse_color),
        (f"Labels: {'ON' if show_labels else 'OFF'}", (255, 255, 255)),
        ("", (255, 255, 255)),
        ("Controls:", (150, 150, 150)),
        ("  q - Quit", (150, 150, 150)),
        ("  m - Toggle mouse", (150, 150, 150)),
        ("  l - Toggle labels", (150, 150, 150)),
        ("  c - Click", (150, 150, 150)),
    ]

    y_offset = panel_y + 22
    for text, color in info_lines:
        cv2.putText(
            image,
            text,
            (panel_x + 15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
        y_offset += 16


def draw_crosshair(image, x: int, y: int, size: int = 20, color=(0, 255, 0)) -> None:
    """
    Draw a crosshair at specified position.

    Args:
        image: Image to draw on
        x: X coordinate
        y: Y coordinate
        size: Crosshair size
        color: Crosshair color
    """
    # Horizontal line
    cv2.line(image, (x - size, y), (x + size, y), color, 2)
    # Vertical line
    cv2.line(image, (x, y - size), (x, y + size), color, 2)
    # Circle
    cv2.circle(image, (x, y), size // 2, color, 1)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """Clamp value between min and max"""
    return max(min_value, min(max_value, value))


def smooth_value(current: float, target: float, factor: float) -> float:
    """
    Apply exponential smoothing.

    Args:
        current: Current value
        target: Target value
        factor: Smoothing factor (0.0-1.0)

    Returns:
        Smoothed value
    """
    return current * (1 - factor) + target * factor
