"""
MediaPipe Hand Gesture Recognition
Supports multiple hands (up to 2) with FPS display
"""

import cv2
import mediapipe as mp
from mediapipe import Image, ImageFormat
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time


# Gesture mapping
GESTURE_MAP = {
    "None": "???",
    "Closed_Fist": "Fist",
    "Open_Palm": "Open Palm",
    "Pointing_Up": "Pointing Up",
    "Thumb_Down": "Thumb Down",
    "Thumb_Up": "Thumb Up",
    "Victory": "Victory",
    "ILoveYou": "ILoveYou",
}


class HandGestureRecognizer:
    """Gesture recognizer - supports up to 2 hands"""

    def __init__(self):
        # Use GestureRecognizer with num_hands=2
        base_options = python.BaseOptions(model_asset_path="gesture_recognizer.task")
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            running_mode=vision.RunningMode.VIDEO,
        )

        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        # Import drawing utilities
        self.mp_drawing = vision.drawing_utils
        self.mp_drawing_styles = vision.drawing_styles

    def recognize(self, frame_rgb, frame_idx):
        """Recognize gestures from frame"""
        mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
        result = self.recognizer.recognize_for_video(mp_image, frame_idx)

        gestures = []
        landmarks = []
        handedness_list = []

        if result.hand_landmarks:
            for i, hand_landmarks in enumerate(result.hand_landmarks):
                landmarks.append(hand_landmarks)

                # Get handedness (Left/Right)
                hand_type = "Unknown"
                if result.handedness and len(result.handedness) > i:
                    raw_hand_type = result.handedness[i][0].category_name
                    # Fix: In selfie mode (flipped), swap Left/Right
                    hand_type = "Right" if raw_hand_type == "Left" else "Left"
                handedness_list.append(hand_type)

                # Get gesture
                gesture = "None"
                confidence = 0
                if result.gestures and len(result.gestures) > i:
                    gesture_category = result.gestures[i][0]
                    gesture = gesture_category.category_name
                    confidence = gesture_category.score

                gesture_text = GESTURE_MAP.get(gesture, gesture)
                gestures.append(
                    {
                        "gesture": gesture,
                        "text": gesture_text,
                        "confidence": confidence,
                    }
                )

        return landmarks, gestures, handedness_list

    def draw_landmarks(self, image, landmarks_list, gestures_list, handedness_list):
        """Draw landmarks and gesture labels"""
        h, w, _ = image.shape

        for i, hand_landmarks in enumerate(landmarks_list):
            # Draw hand landmarks and connections
            self.mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                vision.HandLandmarksConnections.HAND_CONNECTIONS,
                self.mp_drawing_styles.get_default_hand_landmarks_style(),
                self.mp_drawing_styles.get_default_hand_connections_style(),
            )

            # Get wrist position for label
            wrist = hand_landmarks[0]
            x = int(wrist.x * w)
            y = int(wrist.y * h) - 30

            # Draw gesture label
            if i < len(gestures_list):
                # Get handedness and gesture
                hand_type = (
                    handedness_list[i] if i < len(handedness_list) else "Unknown"
                )
                gesture_info = gestures_list[i]
                gesture_text = f"{hand_type}: {gesture_info['text']}"
                confidence = gesture_info["confidence"]

                # Full label with confidence
                full_text = f"{gesture_text} ({confidence:.0%})"

                # Background
                (text_w, text_h), _ = cv2.getTextSize(
                    full_text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2
                )
                cv2.rectangle(
                    image,
                    (x - text_w // 2 - 10, y - text_h - 10),
                    (x + text_w // 2 + 10, y + 10),
                    (0, 0, 0),
                    -1,
                )

                # Text color based on confidence
                if confidence > 0.8:
                    text_color = (0, 255, 0)  # Green
                elif confidence > 0.5:
                    text_color = (0, 255, 255)  # Yellow
                else:
                    text_color = (0, 165, 255)  # Orange

                # Text
                cv2.putText(
                    image,
                    full_text,
                    (x - text_w // 2, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    text_color,
                    2,
                    cv2.LINE_AA,
                )


def draw_info_panel(image, fps, frame_count, hand_count, width, height):
    """Draw info panel on the left side"""
    # Background panel
    panel_x, panel_y = 10, 10
    panel_w, panel_h = 200, 130

    # Draw semi-transparent background
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

    # Info text
    info_lines = [
        ("MediaPipe Hand Gesture", (0, 255, 255)),
        ("", (255, 255, 255)),
        (f"FPS: {fps:.1f}", (0, 255, 0) if fps > 20 else (0, 0, 255)),
        (f"Resolution: {width}x{height}", (255, 255, 255)),
        (f"Hands: {hand_count}", (255, 255, 0)),
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
            0.6,
            color,
            1,
            cv2.LINE_AA,
        )
        y_offset += 20


def main():
    """Main function"""
    recognizer = HandGestureRecognizer()

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Get actual resolution
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("=" * 50)
    print("MediaPipe Hand Gesture Recognition (2 Hands)")
    print("=" * 50)
    print(f"Resolution: {width}x{height}")
    print("Supported gestures:")
    print("  Fist, Open Palm, Thumb Up, Thumb Down")
    print("  Pointing Up, Victory, ILoveYou")
    print("=" * 50)
    print("Press 'q' to exit")
    print("=" * 50)

    # FPS calculation
    frame_idx = 0
    fps = 0
    frame_count = 0
    start_time = time.time()
    fps_update_interval = 10  # Update FPS every 10 frames

    while True:
        success, frame = cap.read()
        if not success:
            print("Error: Cannot read frame")
            break

        # Flip horizontally (selfie mode)
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Recognize gesture
        landmarks, gestures, handedness = recognizer.recognize(frame_rgb, frame_idx)

        # Draw
        if landmarks:
            recognizer.draw_landmarks(frame, landmarks, gestures, handedness)

        # Calculate FPS
        frame_count += 1
        if frame_count >= fps_update_interval:
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            frame_count = 0
            start_time = time.time()

        # Get hand count
        hand_count = len(landmarks)

        # Draw info panel
        draw_info_panel(frame, fps, frame_count, hand_count, width, height)

        # Show image
        cv2.imshow("MediaPipe Hands", frame)

        # Press q to exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_idx += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
