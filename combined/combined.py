import cv2
import numpy as np
import time
import joblib
import os
import copy
import itertools
import threading
from queue import Queue
from collections import deque
import mediapipe as mp
from tensorflow.keras.models import load_model
from action_recognition import process_dynamic_sequence
from utils import extract_keypoints_static, mediapipe_detection
from llama3_translation import get_response

sentence = []
static_prediction_buffer = deque(maxlen=5)
confirmed_static = None
translation_output_queue = Queue()

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
    actions_dynamic = ["your", "my", "what", "hello", "name"]
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
    for lm_group in [
        results.pose_landmarks,
        results.face_landmarks,
        results.left_hand_landmarks,
        results.right_hand_landmarks
    ]:
        if lm_group:
            keypoints.extend([coord for lm in lm_group.landmark for coord in (lm.x, lm.y, lm.z)])
        else:
            keypoints.extend([0] * (33 * 3 if lm_group == results.pose_landmarks else 468 * 3 if lm_group == results.face_landmarks else 21 * 3))
    return np.array(keypoints)


def classify_motion(results, threshold=0.2):
    global prev_keypoints, latest_motion_label
    full_keypoints = results_to_flat_keypoints(results)
    scaled_keypoints = extract_and_scale_hand_keypoints(full_keypoints)
    if scaled_keypoints is None:
        return "Unknown"

    current_keypoints = scaled_keypoints.reshape(42, 3)[:, :2]
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

    if len(movement_cache) < FRAME_WINDOW:
        return "Unknown"

    avg_movement = np.array([[np.mean(movement_cache)]])
    if np.isnan(avg_movement).any():
        return "Unknown"

    probabilities = movement_model.predict_proba(avg_movement)[0]
    prob_dynamic = probabilities[1] 
    # latest_confidence = prob_dynamic * 100 
    # predicted_label = movement_model.predict(avg_movement)[0]
    
    if prob_dynamic >= threshold:
        latest_motion_label = "Dynamic"
        return "Dynamic"
    else:
        latest_motion_label = "Static"
        return "Static"


def process_static_frame(frame, results):
    global result, static_prediction_buffer, confirmed_letter
    if static_model is None:
        return
    hand_landmarks = extract_keypoints_static(results)
    if np.any(hand_landmarks):
        keypoints = calc_landmark_list(frame, hand_landmarks)
        processed_keypoints = pre_process_landmark(keypoints)
        pred_probs = static_model.predict(np.array([processed_keypoints]))[0]
        pred_idx = np.argmax(pred_probs)
        predicted_letter = actions[pred_idx]

        # Check if the prediction is stable
        # confirm if the same prediction is made for 5 consecutive frames
        static_prediction_buffer.append(predicted_letter)

        # Check if all predictions in buffer are the same
        if len(static_prediction_buffer) == static_prediction_buffer.maxlen:
            if all(p == predicted_letter for p in static_prediction_buffer):
                if confirmed_letter != predicted_letter:
                    confirmed_letter = predicted_letter
                    result = confirmed_letter
                    # place in queue for translation
                    translation_queue.put(confirmed_letter)


def process_dynamic_frames(frame_sequence):
    global result, hold_timer
    if lstm_model is None:
        return
    action_result = process_dynamic_sequence(lstm_model, actions_dynamic, frame_sequence, pca)
    result = action_result
    translation_queue.put(action_result)
    # hold_timer = 5



# Thread-safe queue for prediction jobs
prediction_queue = Queue()
translation_queue = Queue()


def static_worker():
    while True:
        job = prediction_queue.get()
        if job is None:
            break
        if isinstance(job, dict) and 'type' in job and job['type'] == 'static':
            frame = job['frame']
            results = job['data']
            process_static_frame(frame, results)
        prediction_queue.task_done()


def dynamic_worker():
    while True:
        job = prediction_queue.get()
        if job is None:
            break
        if isinstance(job, dict) and 'type' in job and job['type'] == 'dynamic':
            process_dynamic_frames(job['data'])
        prediction_queue.task_done()
        
        
def translation_worker(queue, translation_output_queue):
    sentence = []
    last_updated = time.time()
    timeout_duration = 3  # seconds of inactivity = sentence end

    while True:
        time.sleep(0.1)  # frequent check
        if not queue.empty():
            while not queue.empty():
                new_word = queue.get()
                sentence.append(new_word)
                last_updated = time.time()

        if sentence and (time.time() - last_updated > timeout_duration):
            prompt = ' '.join(sentence)
            response = get_response(prompt)
            print("🧠 Prompt:", prompt)
            print("🔠 Interpreted phrase:", response)
            translation_output_queue.put(response)
            sentence.clear()
            


def main():
    global hold_timer
    cap = cv2.VideoCapture(0)
    frame_sequence = []
    # holistic = mp.solutions.holistic.Holistic(
    #     static_image_mode=False,
    #     model_complexity=1,
    #     smooth_landmarks=True,
    #     enable_segmentation=False,
    #     min_detection_confidence=0.5,
    #     min_tracking_confidence=0.5
    # )
    holistic = mp.solutions.holistic.Holistic(min_detection_confidence=0.5,
                                            min_tracking_confidence=0.5)

    # Start background workers
    threading.Thread(target=static_worker, daemon=True).start()
    threading.Thread(target=dynamic_worker, daemon=True).start()
    threading.Thread(target=translation_worker, args=(translation_queue, translation_output_queue), daemon=True).start()

    interpreted_phrase = ""

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not capture frame.")
            break

        image, results = mediapipe_detection(frame, holistic)
        motion_status = classify_motion(results, threshold=0.025)
        frame_sequence.append(frame)
        frame_sequence = frame_sequence[-FRAME_WINDOW:]

        if motion_status == "Static" and len(frame_sequence) == FRAME_WINDOW:
            prediction_queue.put({"type": "static", "frame": image, "data": results})
        elif motion_status == "Dynamic" and len(frame_sequence) == FRAME_WINDOW:
            prediction_queue.put({"type": "dynamic", "data": list(frame_sequence)})  # pass a copy

        if hold_timer > 0:
            hold_timer -= 1

        if not translation_output_queue.empty():
            interpreted_phrase = translation_output_queue.get()

        cv2.putText(frame, f"Gesture: {result}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Motion: {latest_motion_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(frame, f"Interpretation: {interpreted_phrase}", (10, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Motion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    prediction_queue.put(None)  # signal threads to stop
    prediction_queue.put(None)

if __name__ == '__main__':
    main()
