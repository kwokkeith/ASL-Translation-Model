import cv2
import numpy as np
import time
import joblib
import os
import copy
import itertools
import mediapipe as mp
from distance_model import detect_motion_status  # Function to determine Static or Dynamic
# from static_model import process_static_frame  # Function for static gesture recognition
from action_recognition import process_dynamic_sequence  # Function for action recognition
from utils import build_lstm_model, build_static_model, extract_keypoints, extract_keypoints_static, mediapipe_detection  # Import the LSTM model builder
from tensorflow.keras.models import load_model

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
        return n / (max_value+1)

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


def process_static_frame(frame, static_model):
    output_frame = frame.copy()
    # Make detections
    # Initialize Mediapipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    image, results = mediapipe_detection(output_frame, holistic)
    hand_landmarks = extract_keypoints_static(results)

    if np.any(hand_landmarks):
        keypoints = calc_landmark_list(image, hand_landmarks)
        processed_keypoints = pre_process_landmark(keypoints)

        # Perform prediction (per-frame classification)
        pred_probs = static_model.predict(np.array([processed_keypoints]))[0]
        # print("Predicted probabilities:", pred_probs)
        pred_idx = np.argmax(pred_probs)
        predicted_letter = actions[pred_idx]
        confidence = pred_probs[pred_idx]

        return predicted_letter
    return ""


# Constants
FRAME_WINDOW = 15
PCA_ENABLED = True  # Set to True if PCA is enabled
PCA_PATH = "./model/pca_model.pkl"  # Update with actual PCA path
LSTM_MODEL_PATH = "./model/lstm_model.h5"  # Path to the trained dynamic gesture model
STATIC_MODEL_PATH = "./model/static_model.h5"
STATIC_MODEL_DATA = "./data/mp_data_static"
DYNAMIC_MODEL_DATA = "./data/mp_data_dynamic"

# Load PCA model if enabled
pca = None
if PCA_ENABLED:
    try:
        print(f"Loading PCA model from: {PCA_PATH}")
        pca = joblib.load(PCA_PATH)
    except Exception as e:
        print(f"Error loading PCA model: {e}")

# Load the LSTM Model
try:
    print(f"Loading LSTM model from: {LSTM_MODEL_PATH}")
    input_shape = (15, 10)  # Ensure this matches the training input
    num_classes = 5  # Adjust based on the dataset

    # Load dataset actions
    actions_dynamic = np.array([folder for folder in os.listdir(
        DYNAMIC_MODEL_DATA) if os.path.isdir(os.path.join(DYNAMIC_MODEL_DATA, folder))])
    num_classes_dynamic = len(actions_dynamic)

    lstm_model = load_model(LSTM_MODEL_PATH)
except Exception as e:
    print(f"Error loading LSTM model: {e}")
    lstm_model = None

# Load the static Model
try:
    print(f"Loading static model from: {STATIC_MODEL_PATH}")
    input_shape = (42, )  # Ensure this matches the training input
    actions = sorted([
        folder for folder in os.listdir(STATIC_MODEL_DATA)
        if os.path.isdir(os.path.join(STATIC_MODEL_DATA, folder))
    ])
    num_classes = len(actions)
    
    # static_model = build_static_model(num_classes=num_classes)
    static_model = load_model(STATIC_MODEL_PATH)
except Exception as e:
    print(f"Error loading static model: {e}")
    lstm_model = None

# Open Webcam
cap = cv2.VideoCapture(0)
frame_sequence = []
result = ""

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not capture frame.")
        break
    
    # Step 1: Classify Motion Type
    motion_status = detect_motion_status(frame)  # Returns "Static" or "Dynamic"
    frame_sequence.append(frame)
    if len(frame_sequence) > FRAME_WINDOW:
        frame_sequence = []
    
    if motion_status == "Static":
        # Step 2: Process Single Frame with Static Model
        if len(frame_sequence) == FRAME_WINDOW and static_model:
            static_result = process_static_frame(frame, static_model)
            print(f"Static Gesture: {static_result}")
            cv2.putText(frame, f"Motion: {static_result}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            result = static_result
            pass
    
    else:  # "Dynamic"
        # Step 3: Store 30 Frames for Action Recognition
        if len(frame_sequence) == FRAME_WINDOW and lstm_model:
            action_result = process_dynamic_sequence(lstm_model, actions_dynamic, frame_sequence, pca)
            print(f"Detected Action: {action_result}")
            cv2.putText(frame, f"Motion: {action_result}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            result = action_result
    
    # Display Motion Type
    # cv2.putText(frame, f"Motion: {motion_status}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show Video Feed
    cv2.putText(frame, f"Motion: {result}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Motion Detection", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()