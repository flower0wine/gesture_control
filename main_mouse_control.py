"""
Mouse Control Demo
Control mouse cursor using hand tracking (index finger tip)
"""

import cv2
import sys
import time

# Add parent directory to path for imports
sys.path.insert(0, ".")

from hand_tracker import HandDetector, TrackingResult
from hand_tracker.mouse_controller import MouseController
from hand_tracker.utils import FPSCounter, draw_info_panel, draw_crosshair


def main():
    """Main function"""

    # Initialize hand detector
    print("Initializing hand detector...")
    detector = HandDetector(
        num_hands=2,
        model_path="hand_landmarker.task",
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize mouse controller
    print("Initializing mouse controller...")
    mouse = MouseController(
        smoothing=0.3,  # Adjust for responsiveness
        offset_x=0,
        offset_y=0,
        enabled=True,
    )

    # Open camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Get screen size for mouse mapping
    import pyautogui

    screen_width, screen_height = pyautogui.size()
    print(f"Screen size: {screen_width}x{screen_height}")
    print(f"Frame size: {frame_width}x{frame_height}")

    # Initialize FPS counter
    fps_counter = FPSCounter(update_interval=10)

    # State
    frame_idx = 0
    show_labels = False
    last_click_time = 0
    click_cooldown = 0.3  # Seconds between clicks

    print("\n" + "=" * 50)
    print("Hand Tracking Mouse Control")
    print("=" * 50)
    print("Controls:")
    print("  q - Quit")
    print("  m - Toggle mouse control")
    print("  l - Toggle landmark labels")
    print("  c - Click")
    print("=" * 50 + "\n")

    while True:
        # Read frame
        success, frame = cap.read()
        if not success:
            print("Error: Cannot read frame")
            break

        # Flip horizontally (selfie mode)
        frame = cv2.flip(frame, 1)

        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        result = detector.detect(frame_rgb, frame_idx)

        # Update mouse position if enabled
        if mouse.is_enabled and result.hand_count > 0:
            # Use first detected hand
            hand = result.get_hand_by_index(0)
            if hand:
                mouse.update(hand)

                # Get finger position for crosshair
                finger = hand.index_finger_tip
                finger_x = int(finger.x * frame_width)
                finger_y = int(finger.y * frame_height)

                # Draw crosshair at finger position
                draw_crosshair(frame, finger_x, finger_y)

                # Check for click gesture (thumb touches index)
                if mouse.is_clicking(hand):
                    current_time = time.time()
                    if current_time - last_click_time > click_cooldown:
                        mouse.click()
                        last_click_time = current_time
                        # Visual feedback
                        cv2.circle(frame, (finger_x, finger_y), 30, (0, 255, 255), -1)

        # Draw landmarks
        detector.draw_landmarks(frame, result, show_labels)

        # Update FPS
        fps = fps_counter.update()

        # Draw info panel
        draw_info_panel(
            frame,
            fps,
            frame_width,
            frame_height,
            result.hand_count,
            mouse.is_enabled,
            show_labels,
        )

        # Show frame
        cv2.imshow("Hand Tracking Mouse Control", frame)

        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("m"):
            new_state = mouse.toggle()
            print(f"Mouse control: {'ON' if new_state else 'OFF'}")
        elif key == ord("l"):
            show_labels = not show_labels
            print(f"Labels: {'ON' if show_labels else 'OFF'}")
        elif key == ord("c"):
            mouse.click()
            print("Click!")

        frame_idx += 1

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\nExited successfully")


if __name__ == "__main__":
    main()
