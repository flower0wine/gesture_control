"""
Demo script for hand tracking and gesture recognition.

Run this script to test:
- Hand landmark detection
- Finger state detection (extended/curled)
- Swipe gesture detection
- Static gesture recognition (peace, point, fist, etc.)

Press 'q' to quit.
"""

import cv2

from src.core.config import TrackerConfig
from src.gesture import GestureRecognizer
from src.tracking import HandTracker
from src.visualization import HandDrawer


def draw_text_with_background(
    frame,
    text: str,
    position: tuple[int, int],
    font_scale: float = 0.7,
    text_color: tuple[int, int, int] = (0, 255, 0),
    bg_color: tuple[int, int, int] = (0, 0, 0),
) -> None:
    """Draw text with background rectangle."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, thickness
    )
    x, y = position
    cv2.rectangle(
        frame,
        (x - 5, y - text_height - 5),
        (x + text_width + 5, y + baseline + 5),
        bg_color,
        -1,
    )
    cv2.putText(frame, text, (x, y), font, font_scale, text_color, thickness)


def main() -> None:
    """Run hand tracking and gesture recognition demo."""
    # Initialize components
    config = TrackerConfig(max_hands=1)
    tracker = HandTracker(config)
    drawer = HandDrawer()
    recognizer = GestureRecognizer(
        enable_swipe=True,
        enable_static=True,
        swipe_priority=True,
        swipe_min_distance=0.015,  # More sensitive
        swipe_smoothing=0.2,
    )

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    print("=" * 50)
    print("Hand Tracking & Gesture Recognition Demo")
    print("=" * 50)
    print("Features:")
    print("  - Swipe detection (move finger quickly)")
    print("  - Static gestures (peace, point, fist, etc.)")
    print("  - Finger state visualization")
    print("-" * 50)
    print("Press 'q' to quit")
    print("=" * 50)

    timestamp = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Cannot read frame")
            break

        # Flip horizontally for selfie view
        frame = cv2.flip(frame, 1)

        # Detect hands
        hands = tracker.process_frame(frame, timestamp, flipped=True)
        timestamp += 30  # ~30ms per frame

        # Process each detected hand
        for hand in hands:
            # Draw landmarks
            drawer.draw_landmarks(frame, hand)

            # Recognize gesture
            result = recognizer.recognize(hand)

            # Draw hand info
            hand_info = f"Hand: {hand.handedness} ({hand.handedness_score:.2f})"
            draw_text_with_background(frame, hand_info, (10, 30))

            # Draw gesture result
            if result.is_valid:
                gesture_text = f"Gesture: {result.gesture.to_display_name()}"
                confidence_text = f"Confidence: {result.confidence:.2f}"
                draw_text_with_background(frame, gesture_text, (10, 65))
                draw_text_with_background(frame, confidence_text, (10, 100))

                # Draw swipe info if applicable
                if result.gesture.is_swipe:
                    swipe_text = f"Velocity: {result.velocity:.2f} | {result.direction}"
                    draw_text_with_background(
                        frame, swipe_text, (10, 135), font_scale=0.6
                    )
            else:
                draw_text_with_background(frame, "Gesture: Unknown", (10, 65))

            # Draw finger states
            finger_states = recognizer.get_finger_states(hand)
            finger_text = " | ".join(
                f"{k[0].upper()}: {'↑' if v else '↓'}" for k, v in finger_states.items()
            )
            draw_text_with_background(frame, finger_text, (10, 170), font_scale=0.5)

        # Show frame
        cv2.imshow("Hand Tracking & Gesture Demo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    print("\nDemo ended.")


if __name__ == "__main__":
    main()
