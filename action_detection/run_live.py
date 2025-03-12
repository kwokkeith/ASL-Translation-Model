import os
import cv2
import numpy as np
import argparse
import time
from utils import mediapipe_detection, draw_styled_landmarks, \
    extract_keypoints, build_model, extract_optimizer_from_path, \
    find_first_available_camera
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, \
    SpatialDropout1D, BatchNormalization
from llama2_translation import get_response
import threading
import queue
import joblib

# Initialize Mediapipe
mp_holistic = mp.solutions.holistic


## This translation worker triggers based on fixed time interval
def translation_worker(interval, sentence, lock, processed_flag):
    while True:
        time.sleep(interval)  # Wait for the specified interval
        with lock:
            if sentence:
                prompt = ' '.join(sentence)
                response = get_response(prompt)
                print(response)
                processed_flag[0] = True  # Mark as processed



def get_model_input_shape(weights_path):
    """Loads the model architecture and extracts its input shape."""

    # Define your model architecture (must match the original training structure)
    """
    model = Sequential([
        LSTM(64, return_sequences=True, activation='relu',
             input_shape=(30, 1662)),  # Use default shape
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        # Adjust number of actions (classes) accordingly
        Dense(5, activation='softmax')
    ])
    """

    model = Sequential()
    model.add(Input(shape=(15, 1662)))

    model.add(SpatialDropout1D(0.3))

    model.add(LSTM(64,
                   return_sequences=True,
                   recurrent_dropout=0.25,
                   dropout=0.3,
                   activation='relu'))
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
    model.add(Dense(5, activation='softmax'))
    
    # Load the weights
    model.load_weights(weights_path)

    # Extract input shape
    # (sequence_length, fature_length)
    return model.input_shape[1], model.input_shape[2]


def main():
    parser = argparse.ArgumentParser(
        description="Run LSTM action recognition on live video feed")
    parser.add_argument("--model", type=str, required=True,
                        help="Path to the trained model (.h5 file)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset folder containing data")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Confidence threshold for predictions")
    parser.add_argument("--rate", type=float, default=0.001,
                        help="Learning rate of the model loaded")
    parser.add_argument("--mpdc", type=float, default=0.5,
                        help="Minimum detection confidence mediapipe model")
    parser.add_argument("--mptc", type=float, default=0.5,
                        help="Minimum tracking confidence mediapipe model")
    parser.add_argument("--freeze", type=int, default=10,
                        help="Number of predictions before a prediction is valid and displayed")
    parser.add_argument("--sentences", type=int, default=5,
                        help="Number of actions to keep and display")
    parser.add_argument("--interval", type=int, default=5,
                        help="Time interval for sentences to be identified and passed")
    parser.add_argument("--pca_enabled", type=bool, required=True,
                        help="Toggles if PCA was used")
    parser.add_argument("--pca_path", type=str, default=None,
                        help="Path to saved PCA model (.pkl)")

    args = parser.parse_args()

    # Load dataset actions
    actions = np.array([folder for folder in os.listdir(
        args.dataset) if os.path.isdir(os.path.join(args.dataset, folder))])
    num_classes = len(actions)

    # Load PCA model if enabled
    pca = None
    if args.pca_enabled:
        if args.pca_path is None:
            print("Error: PCA is enabled but no path provided for the PCA model.")
            return
        print(f"Loading PCA model from: {args.pca_path}")
        pca = joblib.load(args.pca_path)

    # Load LSTM Model
    model = load_model(args.model)
    print(f"Loaded model weights from: {args.model}")

    # Initialize sequence storage
    sequence = []
    sentence = []
    predictions = []
    lock = threading.Lock()
    processed_flag = [False]
    threshold = args.threshold

    # Start translation worker thread
    worker_thread = threading.Thread(target=translation_worker, args=(args.interval, sentence, lock, processed_flag))
    worker_thread.daemon = True
    worker_thread.start()

    # Find the first available camera
    camera_index = find_first_available_camera()
    if camera_index is None:
        print("Unable to find a camera that is available")
        return

    # Open webcam
    cap = cv2.VideoCapture(camera_index)

    with mp_holistic.Holistic(min_detection_confidence=args.mpdc,
                              min_tracking_confidence=args.mptc) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Camera feed not available.")
                break

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)

            # Extract keypoints
            keypoints = extract_keypoints(results)

            # Apply PCA if enabled
            if args.pca_enabled and pca is not None:
                keypoints = keypoints.reshape(1, -1)  # Flatten to (1, feature_dim)
                keypoints = pca.transform(keypoints)  # Apply PCA
                keypoints = keypoints.flatten()  # Convert back to 1D array

            sequence.append(keypoints)

            # Keep only the last `sequence_length` frames
            sequence = sequence[-15:]

            # Only run prediction when we have enough frames
            if len(sequence) == 15:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predicted_action = actions[np.argmax(res)]
                confidence = res[np.argmax(res)]

                predictions.append(np.argmax(res))

                # Check if action is stable across last `freeze` frames
                if len(predictions) > args.freeze and \
                        np.unique(predictions[-args.freeze:])[0] == np.argmax(res):
                    if confidence > threshold:
                        if len(sentence) == 0 or \
                                predicted_action != sentence[-1]:
                            sentence.append(predicted_action)

                if len(sentence) > args.sentences:
                    sentence = sentence[-args.sentences:]

                if processed_flag[0]:
                    with lock:
                        sentence.clear()
                        processed_flag[0] = False  # Reset flag

                # Display action and confidence
                cv2.putText(image, f"Action: {predicted_action}",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, f"Confidence: {confidence:.2f}",
                            (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)

            # Show video feed
            cv2.imshow('Action Recognition', image)

            # Quit with 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
