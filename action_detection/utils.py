import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam, AdamW, SGD

# Initialize Mediapipe modules
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def find_first_available_camera(max_ports=10):
    """Finds the first available camera index."""
    for index in range(max_ports):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            cap.release()
            return index  # Return the first working camera index
    return None  # No available cameras found


def mediapipe_detection(image, model):
    """Processes an image using the Mediapipe model."""
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image.flags.writeable = False  # Improve performance
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Restore
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR
    return image, results


def draw_styled_landmarks(image, results):
    """ To draw the landmarks from mediapipe and use different colors """
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks,
                              mp_holistic.FACEMESH_CONTOURS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 110, 10),
                                  thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(
                                  color=(80, 256, 121),
                                  thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks,
                              mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(80, 22, 10),
                                  thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(80, 44, 121),
                                  thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(121, 22, 76),
                                  thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(121, 44, 250),
                                  thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(
                                  color=(245, 117, 66),
                                  thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(
                                  color=(245, 66, 230),
                                  thickness=2, circle_radius=2)
                              )


def extract_keypoints(results):
    """Extract Keypoints from mediapipe in form of np array"""
    pose = np.array([[res.x, res.y, res.z, res.visibility]
                     for res in results.pose_landmarks.landmark]) \
        .flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z]
                     for res in results.face_landmarks.landmark]) \
        .flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z]
                   for res in results.left_hand_landmarks.landmark]) \
        .flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z]
                   for res in results.right_hand_landmarks.landmark]) \
        .flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


def build_model(input_shape, num_classes):
    """Creates the LSTM model architecture."""
    model = Sequential([
        LSTM(64, return_sequences=True,
             activation='relu', input_shape=input_shape),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model


def extract_optimizer_from_path(model_path, learning_rate=0.001):
    """Extracts optimizer type from the model path string."""
    if "AdamW" in model_path:
        return AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    elif "SGD" in model_path:
        return SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        return Adam(learning_rate=learning_rate)  # Default to Adam
