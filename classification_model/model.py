import cv2
import mediapipe as mp
import numpy as np
import joblib
from collections import deque
import time

#LOAD MODEL
model = joblib.load("./training_utils/logistic_regression_movement_exponential.joblib")

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
WEIGHTING = "exponential"

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


def extract_and_scale_hand_keypoints(full_keypoints, target_size=100):
    """Extracts hand keypoints and scales them to a target_size x target_size box."""
    left_hand_start = 1536
    right_hand_start = left_hand_start + 63

    left_hand = full_keypoints[left_hand_start:left_hand_start + 63]
    right_hand = full_keypoints[right_hand_start:right_hand_start + 63]

    if len(left_hand) != 63:
        left_hand = np.zeros(63)
    if len(right_hand) != 63:
        right_hand = np.zeros(63)

    hand_keypoints = np.concatenate([left_hand, right_hand])  # Shape (126,)
    hand_keypoints_reshaped = hand_keypoints.reshape(2, 21, 3)
    xy_keypoints = hand_keypoints_reshaped[:, :, :2]

    min_xy = np.min(xy_keypoints, axis=(0, 1))
    max_xy = np.max(xy_keypoints, axis=(0, 1))
    wh = max_xy - min_xy

    if np.any(wh == 0):
        return None

    scale = target_size / np.max(wh)
    xy_keypoints[:, :, 0] = (xy_keypoints[:, :, 0] - min_xy[0]) * scale
    xy_keypoints[:, :, 1] = (xy_keypoints[:, :, 1] - min_xy[1]) * scale

    hand_keypoints_reshaped[:, :, :2] = xy_keypoints
    return hand_keypoints_reshaped.reshape(-1)

def results_to_flat_keypoints(results):
    keypoints = []
    for lm_group in [
        results.pose_landmarks,
        results.face_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks
    ]:
        if lm_group:
            keypoints.extend([coord for lm in lm_group.landmark for coord in (lm.x, lm.y, lm.z)])
        else:
            if lm_group == results.pose_landmarks:
                keypoints.extend([0] * (33 * 3))
            elif lm_group == results.face_landmarks:
                keypoints.extend([0] * (468 * 3))
            else:
                keypoints.extend([0] * (21 * 3))
    return np.array(keypoints)

def process_frame(results):
    global prev_keypoints
    full_keypoints = results_to_flat_keypoints(results)
    scaled_keypoints = extract_and_scale_hand_keypoints(full_keypoints)
    if scaled_keypoints is None:
        return False

    current_keypoints = scaled_keypoints.reshape(42, 3)[:, :2]  # Use only x, y
    hand_count = 0
    if results.left_hand_landmarks:
        hand_count += 1
    if results.right_hand_landmarks:
        hand_count += 1
    movement_magnitude = 0
    if prev_keypoints is not None:
        movement_magnitude = np.linalg.norm(current_keypoints - prev_keypoints, axis=1)
        movement_magnitude = np.nanmean(movement_magnitude)

        if hand_count == 1:
            movement_magnitude *= 2
        movement_cache.append(movement_magnitude)

    prev_keypoints = current_keypoints.copy()
    return True


def predict_gesture(threshold=0.18):
    global latest_prediction, latest_confidence
    if len(movement_cache) < sequence_length:
        return

    avg_movement = np.array([[np.mean(movement_cache)]])
    if np.isnan(avg_movement).any():
        return

    probabilities = model.predict_proba(avg_movement)[0]
    prob_dynamic = probabilities[1]  
    latest_confidence = prob_dynamic * 100

    if prob_dynamic >= threshold:
        latest_prediction = "Dynamic"
    else:
        latest_prediction = "Static"


    
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

        if results.left_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        if results.right_hand_landmarks:
            mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        cv2.putText(frame, f"Prediction: {latest_prediction} ({latest_confidence:.2f}%)", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Avg Movement: {np.mean(movement_cache) if movement_cache else 0:.4f}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow("Live Gesture Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
