import cv2
import numpy as np
import time
import joblib
import os
import copy
import itertools
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, f1_score
from utils import extract_keypoints_static, mediapipe_detection

# Constants
FRAME_WINDOW = 15
PCA_ENABLED = True
LOGISTIC_MODEL_PATH = "./model/logistic_regression_movement.joblib"
PCA_PATH = "./model/pca_model.pkl"
LSTM_MODEL_PATH = "./model/lstm_model.h5"
STATIC_MODEL_PATH = "./model/static_model.h5"
STATIC_MODEL_DATA = "./data/mp_data_static"
DYNAMIC_MODEL_DATA = "./data/mp_data_dynamic"

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load PCA model if enabled
pca = None
if PCA_ENABLED:
    try:
        print(f"Loading PCA model from: {PCA_PATH}")
        pca = joblib.load(PCA_PATH)
    except Exception as e:
        print(f"Error loading PCA model: {e}")

# Load movement classifier
print(f"Loading logistic regression model from: {LOGISTIC_MODEL_PATH}")
movement_model = joblib.load(LOGISTIC_MODEL_PATH)

# Movement tracking variables
movement_cache = deque(maxlen=FRAME_WINDOW)
prev_keypoints = None
latest_motion_label = "Unknown"

# Load LSTM Model
lstm_model = None
try:
    print(f"Loading LSTM model from: {LSTM_MODEL_PATH}")
    actions_dynamic = np.array([folder for folder in os.listdir(DYNAMIC_MODEL_DATA) if os.path.isdir(os.path.join(DYNAMIC_MODEL_DATA, folder))])
    print("Action dynamics list:", actions_dynamic)
    lstm_model = load_model(LSTM_MODEL_PATH)
except Exception as e:
    print(f"Error loading LSTM model: {e}")

# Load Static Model
static_model = None
actions = []
try:
    print(f"Loading static model from: {STATIC_MODEL_PATH}")
    actions = sorted([folder for folder in os.listdir(STATIC_MODEL_DATA) if os.path.isdir(os.path.join(STATIC_MODEL_DATA, folder))])
    static_model = load_model(STATIC_MODEL_PATH)
except Exception as e:
    print(f"Error loading static model: {e}")

result = ""  # Shared result variable
hold_timer = 0 # For dynamic gesture display hold

def pre_process_landmark(landmark_list):
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for landmark in landmarks:
        landmark_x = min(int(landmark[0] * image_width), image_width - 1)
        landmark_y = min(int(landmark[1] * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def extract_and_scale_hand_keypoints(full_keypoints, target_size=100):
    left_hand_start = 1536
    right_hand_start = left_hand_start + 63
    left_hand = full_keypoints[left_hand_start:left_hand_start + 63]
    right_hand = full_keypoints[right_hand_start:right_hand_start + 63]
    if len(left_hand) != 63:
        left_hand = np.zeros(63)
    if len(right_hand) != 63:
        right_hand = np.zeros(63)
    hand_keypoints = np.concatenate([left_hand, right_hand])
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
    # for lm_group in [
    #     results.pose_landmarks,
    #     results.face_landmarks,
    #     results.left_hand_landmarks,
    #     results.right_hand_landmarks
    # ]:
    print(f"results: {results}")
    for lm_group in results:
        if lm_group:
            keypoints.extend([coord for lm in lm_group.landmark for coord in (lm.x, lm.y, lm.z)])
        else:
            keypoints.extend([0] * (33 * 3 if lm_group == results.pose_landmarks else 468 * 3 if lm_group == results.face_landmarks else 21 * 3))
    return np.array(keypoints)


def classify_motion_from_sequence(sequence, threshold=0.2):
    """
    Classifies motion from a sequence of 30 frames.
    Each frame should be a NumPy array of shape (1662,).
    Returns "Static", "Dynamic", or "Unknown" if not enough valid frames.
    """
    if len(sequence) != 30:
        raise ValueError(f"Expected 30 frames, got {len(sequence)}.")

    keypoint_list = []
    valid_frame_indices = []

    for idx, frame in enumerate(sequence):
        scaled = extract_and_scale_hand_keypoints(frame)
        if scaled is None:
            # print(f"[DEBUG] Skipping frame {idx}: invalid or missing hand keypoints.")
            continue

        xy_keypoints = scaled.reshape(42, 3)[:, :2]
        keypoint_list.append(xy_keypoints)
        valid_frame_indices.append(idx)

    # Require at least 3 valid frames to compute motion (2 diffs)
    if len(keypoint_list) < 3:
        print("[DEBUG] Not enough valid frames to classify motion.")
        return "Unknown"

    # Compute movement magnitudes
    movement_magnitudes = []
    for i in range(1, len(keypoint_list)):
        diff = np.linalg.norm(keypoint_list[i] - keypoint_list[i - 1], axis=1)
        avg_diff = np.nanmean(diff)
        if not np.isnan(avg_diff):
            movement_magnitudes.append(avg_diff)
        else:
            print(f"[DEBUG] Skipping movement calc between frames {valid_frame_indices[i-1]} and {valid_frame_indices[i]} due to NaN.")

    if len(movement_magnitudes) == 0:
        print("[DEBUG] All movement magnitude calculations resulted in NaN.")
        return "Unknown"

    avg_movement = np.array([[np.mean(movement_magnitudes)]])
    probabilities = movement_model.predict_proba(avg_movement)[0]
    prob_dynamic = probabilities[1]

    print(f"[DEBUG] Avg movement: {avg_movement[0][0]:.4f}, Prob(Dynamic): {prob_dynamic:.4f}")

    if prob_dynamic >= threshold:
        return "Dynamic"
    else:
        return "Static"


def process_static_frame(frame, results):
    if static_model is None:
        return
    # print(results)
    hand_landmarks = extract_keypoints_static(results)
    if np.any(hand_landmarks):
        keypoints = calc_landmark_list(frame, hand_landmarks)
        processed_keypoints = pre_process_landmark(keypoints)
        pred_probs = static_model.predict(np.array([processed_keypoints]))[0]
        pred_idx = np.argmax(pred_probs)
        predicted_letter = actions[pred_idx]
        result = predicted_letter
        return result
    return ""


def process_dynamic_frames(frame_sequence):
    if lstm_model is None:
        return
    return process_dynamic_sequence(lstm_model, actions_dynamic, frame_sequence, pca)


def process_dynamic_sequence(model, actions, keypoint_frames, pca):
    """
    keypoint_frames: list of 30 np.ndarray, each shape (1662,)
    """
    keypoint_frames = keypoint_frames[-15:]
    sequence = []

    for keypoints in keypoint_frames:
        if pca is not None:
            keypoints = keypoints.reshape(1, -1)
            keypoints = pca.transform(keypoints).flatten()
        sequence.append(keypoints)

    if len(sequence) == 15:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        predicted_action = actions[np.argmax(res)]
        return predicted_action
    else:
        print("Processing dynamic Sequence: sequence length is not 15")
        return 
    return "Unknown"



def classify_dynamic_sequence_from_keypoints(frame_keypoints_sequence):
    motion_status = classify_motion_from_sequence(frame_keypoints_sequence, threshold=0.025)
    frame_sequence = frame_keypoints_sequence[-FRAME_WINDOW:]
    if motion_status == "Static":
        print("Detected Static")
        last_frame = frame_sequence[FRAME_WINDOW - 1]
        right_hand_start = 1536 + 63
        right_hand = last_frame[right_hand_start:right_hand_start + 63]
        if len(right_hand) == 63:
            right_hand_kp = right_hand.reshape(21, 3)[:, :2].tolist()
            processed = pre_process_landmark(right_hand_kp)
            pred_probs = static_model.predict(np.array([processed]))[0]
            pred_idx = np.argmax(pred_probs)
            result = actions[pred_idx]
        else:
            result = "Unknown"
    elif motion_status == "Dynamic":
        print("Detected Dynamic")
        result = process_dynamic_frames(frame_sequence)
    else:
        result = "Unknown"
    # print(f"Prediction: {result}")
    return result  


def load_first_30_frames_from_path(sample_path):
    """
    Given a directory with numbered .npy files, loads the first 30 frames sorted by filename.
    Returns a list of np.ndarray with shape (30, 1662)
    """
    frame_files = sorted([f for f in os.listdir(sample_path) if f.endswith('.npy')],
                        key=lambda x: int(x.split('.')[0]))
    
    if len(frame_files) < 30:
        print(f"Warning: Only {len(frame_files)} frames found in {sample_path}, skipping.")
        return None

    frame_sequence = []
    for fname in frame_files[:30]:
        fpath = os.path.join(sample_path, fname)
        try:
            frame = np.load(fpath)
            if frame.shape[0] != 1662:
                print(f"Invalid shape {frame.shape} in {fpath}")
                return None
            frame_sequence.append(frame)
        except Exception as e:
            print(f"Error loading {fpath}: {e}")
            return None

    return frame_sequence


def infer_ood(base_path="./ood_dataset/mp_data_dynamic_ood"):
    print(f"\nRunning OOD inference on {base_path}...")
    y_true = []
    y_pred = []

    for gesture_label in sorted(os.listdir(base_path)):
        gesture_dir = os.path.join(base_path, gesture_label)
        if not os.path.isdir(gesture_dir):
            continue

        for sample_folder in sorted(os.listdir(gesture_dir)):
            sample_path = os.path.join(gesture_dir, sample_folder)
            if not os.path.isdir(sample_path):
                continue

            sequence = load_first_30_frames_from_path(sample_path)
            if sequence is None:
                continue

            try:
                prediction = classify_dynamic_sequence_from_keypoints(sequence)
                print(f"{gesture_label}/{sample_folder} ➜ Predicted: {prediction}")
                y_true.append(gesture_label)
                y_pred.append(prediction if prediction is not None else "Unknown")
            except Exception as e:
                print(f"Error predicting sample {gesture_label}/{sample_folder}: {e}")
                y_true.append(gesture_label)
                y_pred.append("Unknown")
    return y_true, y_pred



def main():
    y_true_dyn, y_pred_dyn = infer_ood("./ood_dataset/mp_data_dynamic_ood")
    y_true_static, y_pred_static = infer_ood("./ood_dataset/mp_data_static_ood")

    y_true = y_true_dyn + y_true_static
    y_pred = y_pred_dyn + y_pred_static

    print("\n Classification Report:\n")
    print(classification_report(y_true, y_pred, zero_division=0))  # zero_division=0 avoids division by 0 errors

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    print(f"Macro-averaged F1 Score: {macro_f1:.4f}")


if __name__ == '__main__':
    main()
