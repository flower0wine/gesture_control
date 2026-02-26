"""
MediaPipe Hand Landmarks Detection
Only detects hand keypoints (21 points per hand) - no gesture recognition
"""

import cv2
import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


# Landmark names for reference
LANDMARK_NAMES = [
    "WRIST",
    "THUMB_CMC",
    "THUMB_MCP",
    "THUMB_IP",
    "THUMB_TIP",
    "INDEX_FINGER_MCP",
    "INDEX_FINGER_PIP",
    "INDEX_FINGER_DIP",
    "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP",
    "MIDDLE_FINGER_PIP",
    "MIDDLE_FINGER_DIP",
    "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP",
    "RING_FINGER_PIP",
    "RING_FINGER_DIP",
    "RING_FINGER_TIP",
    "PINKY_MCP",
    "PINKY_PIP",
    "PINKY_DIP",
    "PINKY_TIP",
]


class HandLandmarkDetector:
    """Hand landmark detector - gets 21 keypoints per hand"""

    def __init__(self):
        # Use HandLandmarker (keypoints only, no gesture)
        base_options = python.BaseOptions(model_asset_path="hand_landmarker.task")
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )

        self.landmarker = vision.HandLandmarker.create_from_options(options)

        # Import drawing utilities
        self.mp_drawing = vision.drawing_utils
        self.mp_drawing_styles = vision.drawing_styles

    def detect(self, frame_rgb, frame_idx):
        """Detect hand landmarks from frame"""
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self.landmarker.detect_for_video(mp_image, frame_idx)

        landmarks_list = []
        handedness_list = []

        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                landmarks_list.append(hand_landmarks)

                # Get handedness (Left/Right)
                hand_type = "Unknown"
                if result.handedness and len(result.handedness) > i:
                    raw_hand_type = result.handedness[i][0].category_name
                    # Fix: In selfie mode (flipped), swap Left/Right
                    hand_type = "Right" if raw_hand_type == "Left" else "Left"
                handedness_list.append(hand_type)

        return landmarks_list, handedness_list

    def draw_landmarks(self, image, landmarks_list, handedness_list, show_labels=False):
        """Draw landmarks on image"""
        h, w, _ = image.shape

        # Colors for different hands
        colors = [(0, 255, 0), (255, 0, 255)]  # Green for first, Purple for second

        for hand_idx, hand_landmarks in enumerate(landmarks_list):
            color = colors[hand_idx % len(colors)]

            # Draw hand landmarks and connections
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Get handedness
            hand_type = (
                handedness_list[hand_idx]
                if hand_idx < len(handedness_list)
                else "Unknown"
            )

            # Draw hand label at wrist
            wrist = hand_landmarks[0]
            x = int(wrist.x * w)
            y = int(wrist.y * h) - 30

            # Label background
            label = f"{hand_type} Hand"
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            cv2.rectangle(
                image,
                (x - text_w // 2 - 8, y - text_h - 8),
                (x + text_w // 2 + 8, y + 8),
                (0, 0, 0),
                -1,
            )

            # Label text
            cv2.putText(
                image,
                label,
                (x - text_w // 2, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

            # Optionally show landmark numbers
            if show_labels:
                for lm_idx, lm in enumerate(hand_landmarks):
                    lm_x = int(lm.x * w)
                    lm_y = int(lm.y * h)

                    # Draw small circle at each landmark
                    cv2.circle(image, (lm_x, lm_y), 3, color, -1)

                    # Draw landmark index (for first hand only to avoid clutter)
                    if hand_idx == 0 and lm_idx % 4 == 0:  # Show every 4th landmark
                        cv2.putText(
                            image,
                            str(lm_idx),
                            (lm_x + 5, lm_y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA,
                        )

        return image


def draw_info_panel(image, fps, width, height, hand_count):
    """Draw info panel"""
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 220, 120

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

    # Info
    info_lines = [
        ("MediaPipe Hand Landmarks", (0, 255, 255)),
        ("", (255, 255, 255)),
        (f"FPS: {fps:.1f}", (0, 255, 0) if fps > 20 else (0, 0, 255)),
        (f"Resolution: {width}x{height}", (255, 255, 255)),
        (f"Hands: {hand_count}", (255, 255, 0)),
        (f"Landmarks: {hand_count * 21}", (255, 255, 0)),
        ("", (255, 255, 255)),
        ("Press 'q' to exit", (150, 150, 150)),
    ]

    y_offset = panel_y + 25
    for text, color in info_lines:
        cv2.putText(
            image,
            text,
            (panel_x + 15, y_offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            1,
            cv2.LINE_AA,
        )
        y_offset += 18


def main():
    """Main function"""
    detector = HandLandmarkDetector()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 50)
    print("MediaPipe Hand Landmarks Detection")
    print("=" * 50)
    print(f"Resolution: {width}x{height}")
    print("Detects 21 keypoints per hand (42 points for 2 hands)")
    print("No gesture recognition - keypoints only")
    print("=" * 50)
    print("Press 'q' to exit")
    print("=" * 50)

    # FPS calculation
    frame_idx = 0
    fps = 0
    frame_count = 0
    start_time = time.time()

    # Toggle for showing landmark labels
    show_labels = False

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Cannot read frame")
            break

        # Flip horizontally (selfie mode)
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect landmarks
        landmarks_list, handedness_list = detector.detect(frame_rgb, frame_idx)

        # Draw landmarks
        if landmarks_list:
            detector.draw_landmarks(frame, landmarks_list, handedness_list, show_labels)

        # Calculate FPS
        frame_count += 1
        if frame_count >= 10:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # Draw info panel
        hand_count = len(landmarks_list)
        draw_info_panel(frame, fps, width, height, hand_count)

        # Show image
        cv2.imshow("MediaPipe Hand Landmarks", frame)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("l"):  # Toggle labels
            show_labels = not show_labels
            print(f"Labels: {'ON' if show_labels else 'OFF'}")

        frame_idx += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
