import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, \
    SpatialDropout1D, BatchNormalization
from tensorflow.keras.optimizers import Adam, AdamW, SGD
import matplotlib.pyplot as plt
import os
import joblib


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


def get_hand_keypoints(feature_array):
    """Extracts left-hand and right-hand keypoints from the feature vector"""
    lh = feature_array[1536:1599].reshape(21, 3)  # (21, 3)
    rh = feature_array[1599:1662].reshape(21, 3)  # (21, 3)
    return lh, rh


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
    # model = Sequential([
    #     LSTM(64, return_sequences=True,
    #          activation='relu', input_shape=input_shape),
    #     LSTM(128, return_sequences=True, activation='relu'),
    #     LSTM(64, return_sequences=False, activation='relu'),
    #     Dense(64, activation='relu'),
    #     Dense(32, activation='relu'),
    #     Dense(num_classes, activation='softmax')  # Output layer
    # ])

    model = Sequential()
    model.add(SpatialDropout1D(0.3))

    model.add(LSTM(64,
                   return_sequences=True,
                   recurrent_dropout=0.25,
                   dropout=0.3,
                   activation='relu',
                   input_shape=input_shape))
    model.add(BatchNormalization())

    model.add(LSTM(128,
                   return_sequences=True,
                   recurrent_dropout=0.25,
                   dropout=0.3,
                   activation='relu'))
    model.add(BatchNormalization())

    model.add(LSTM(64,
                   return_sequences=False,
                   recurrent_dropout=0.25,
                   dropout=0.3,
                   activation='relu'))
    model.add(BatchNormalization())

    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    # Output layer with softmax activation
    model.add(Dense(num_classes, activation='softmax'))
    
    return model


def extract_optimizer_from_path(model_path, learning_rate=0.001):
    """Extracts optimizer type from the model path string."""
    if "AdamW" in model_path:
        return AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    elif "SGD" in model_path:
        return SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        return Adam(learning_rate=learning_rate)  # Default to Adam


def plot_metric(model_training_history,
                metric_name_1,
                metric_name_2,
                plot_name):
    """
    This function plots the metrics passed to it in a graph.
    Args:
        model_training_history: A history object containing a record of
        and validation loss values and metrics values at successive epochs.
        metric_name_1: The name of the first metric that needs to be plotted
        metric_name_2: The name of the second metric that needs to be plotted.
        plot_name: The title of the graph

    Returns:
        plt: the matplotlib plot object.
    """
    # Get metric values using metric names as identifiers
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Construct range object used as x-axis of the graph
    epochs = range(len(metric_value_1))

    # plot the graph
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)

    # Add title to plot
    plt.title(str(plot_name))

    # Add legends to plot
    plt.legend()

    return plt


def load_pca_model(pca_model_path):
    """Loads the pre-trained PCA model."""
    if os.path.exists(pca_model_path):
        pca = joblib.load(pca_model_path)
        print(f"Loaded PCA model from {pca_model_path}")
        return pca
    else:
        raise FileNotFoundError(f"PCA model not found at {pca_model_path}")

