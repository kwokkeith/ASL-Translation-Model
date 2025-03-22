import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import time

#LOAD MODEL
model = joblib.load("./logistic_regression_movement.joblib")

#init holistic model & drawing utils
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

#
sequence_length = 30
movement_cache = deque(maxlen=sequence_length)

# State variables
prev_keypoints = None
latest_prediction = "Unknown"
latest_confidence = 0.0
frame_count = 0
last_check_time = time.time()
valid_frames_this_second = 0

# Weighting strategy
WEIGHTING = "linear"

def compute_weighted_avg(movements):
    n = len(movements)
    if n == 0:
        return 0
    weights = np.ones(n)
    #higher weightage to recent frames
    if WEIGHTING == "linear":
        weights = np.linspace(1, 0.1, n)
    #highest weightage to recent frames
    elif WEIGHTING == "exponential":
        weights = np.exp(np.linspace(0, -3, n))
    #highest weightage where movements are strongest
    elif WEIGHTING == "peak":
        weights = movements / np.max(movements) if np.max(movements) > 0 else np.ones(n)
    
    weights /= np.sum(weights)
    return np.sum(movements * weights)


def extract_hand_keypoints(results):
    keypoints = np.full((2, 21, 2), np.nan)
    hand_count = 0
    if results.left_hand_landmarks:
        keypoints[0] = np.array([(lm.x, lm.y) for lm in results.left_hand_landmarks.landmark])
        hand_count += 1
    if results.right_hand_landmarks:
        keypoints[1] = np.array([(lm.x, lm.y) for lm in results.right_hand_landmarks.landmark])
        hand_count += 1
    if hand_count == 0:
        return None, 0
    return keypoints.reshape(42, 2), hand_count

def process_frame(results):
    global prev_keypoints
    keypoints, hand_count = extract_hand_keypoints(results)
    if keypoints is None:
        return False
    movement_magnitude = 0
    if prev_keypoints is not None:
        movement_magnitude = np.linalg.norm(keypoints - prev_keypoints, axis=1)
        movement_magnitude = np.nanmean(movement_magnitude)
        if hand_count == 1:
            movement_magnitude *= 2
        movement_cache.append(movement_magnitude)
    prev_keypoints = keypoints.copy()
    return True

def predict_gesture():
    global latest_prediction, latest_confidence
    if len(movement_cache) < sequence_length:
        return
    avg_movement = np.array([[np.mean(movement_cache)]])
    if np.isnan(avg_movement).any():
        return
    probabilities = model.predict_proba(avg_movement)[0]
    predicted_label = model.predict(avg_movement)[0]
    latest_confidence = np.max(probabilities) * 100
    latest_prediction = "Dynamic" if predicted_label == 1 else "Static"

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        detected = process_frame(results)
        if detected:
            valid_frames_this_second += 1
        predict_gesture()

        # If no detection in 0.03 seconds, inject a 0.0
        if time.time() - last_check_time >= 0.03:
            if valid_frames_this_second == 0:
                movement_cache.append(0.0)
            valid_frames_this_second = 0
            last_check_time = time.time()

        # Draw landmarks
        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Display info
        cv2.putText(frame, f"Prediction: {latest_prediction} ({latest_confidence:.2f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Avg Movement: {np.mean(movement_cache) if movement_cache else 0:.4f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Live Gesture Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
