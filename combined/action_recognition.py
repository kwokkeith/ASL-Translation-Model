import os
import cv2
import numpy as np
from utils import mediapipe_detection, draw_styled_landmarks, \
    extract_keypoints, build_model, extract_optimizer_from_path, \
    find_first_available_camera
import mediapipe as mp
from sklearn.decomposition import PCA as pca
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, load_model

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic


def process_dynamic_sequence(model, actions, frames, pca):
    """Processes 30 frames to recognize a dynamic hand action using the given LSTM model."""
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    
    ## Take only the last 15 frames
    frames = frames[:15]
    sequence=[]

    for frame in frames:
        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        
        # Extract keypoints
        keypoints = extract_keypoints(results)
        keypoints = keypoints.reshape(1, -1)  # Flatten to (1, feature_dim)
        keypoints = pca.transform(keypoints)  # Apply PCA
        keypoints = keypoints.flatten()  # Convert back to 1D array

        sequence.append(keypoints)  

    if len(sequence) == 15:
        res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
        predicted_action = actions[np.argmax(res)]
        confidence = res[np.argmax(res)]
        predicted_label = np.argmax(res)
        asl_labels = ["your", "my", "what", "hello", "name"]

        return asl_labels[predicted_label]

    return ""