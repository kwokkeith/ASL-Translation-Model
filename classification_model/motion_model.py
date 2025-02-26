import cv2
import mediapipe as mp
import collections
import numpy as np
import time

# Initialize Mediapipe Hand Model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture Video
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1920)

# Store last 60 frames of hand landmark positions
prev_hand_positions = {}  # Dictionary to store hand positions per hand
hand_motion_threshold = 0.02  # Normalized movement threshold

# Variables for FPS Calculation
prev_time = time.time()
fps_log = []  # Store FPS values

while cap.isOpened():
    start_time = time.time()  # Start time for FPS control

    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Track hand movements
    current_hand_positions = {}

    # Process Hand Landmarks
    if results.multi_hand_landmarks:
        for idx, (hand_landmarks, handedness) in enumerate(zip(results.multi_hand_landmarks, results.multi_handedness)):
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract (x, y) positions of hand landmarks
            hand_positions = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark])

            # Determine Left or Right Hand
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            hand_id = f"{hand_label} Hand"  # Unique ID per hand

            # Store current positions
            current_hand_positions[hand_id] = hand_positions

            # Compute movement magnitude **only if we have a previous position**
            if hand_id in prev_hand_positions:
                prev_positions = prev_hand_positions[hand_id]
                movement = np.linalg.norm(hand_positions - prev_positions, axis=1)
                hand_movement_magnitude = np.mean(movement)  # Average movement

                # Determine if this hand is moving
                is_hand_moving = hand_movement_magnitude > hand_motion_threshold
                motion_status = "Moving" if is_hand_moving else "Static"

                # Display Motion Status & Hand Label
                cv2.putText(frame, f"{hand_id}: {motion_status}", (20, 80 + idx * 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, 
                            (0, 0, 255) if is_hand_moving else (255, 255, 255), 
                            2, cv2.LINE_AA)

    # Update previous hand positions for next frame
    prev_hand_positions = current_hand_positions.copy()

    # Calculate FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
    prev_time = current_time

    # Display FPS
    cv2.putText(frame, f"FPS: {int(fps)}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Display Video Feed
    cv2.imshow("Hand Tracking with Hand Motion Detection", frame)

    # Cap FPS at 30
    time.sleep(max(0, (1/30) - (time.time() - start_time)))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
