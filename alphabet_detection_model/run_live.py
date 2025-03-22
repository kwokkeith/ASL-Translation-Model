import os
import cv2
import numpy as np
import argparse
import mediapipe as mp
import tensorflow as tf
from utils import find_first_available_camera, mediapipe_detection, draw_styled_landmarks, extract_optimizer_from_path, extract_keypoints
from utils1.cvfpscalc import CvFpsCalc
import copy
import itertools
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier


# Global variables for accumulating predicted alphabets
word = ""
consecutive_frames = 0
prev_sign = None

def get_args():
    parser = argparse.ArgumentParser(
        description="Run live alphabet recognition using a trained model")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the model weights (.h5 file)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset folder containing label subdirectories")
    parser.add_argument("--mpdc", type=float, default=0.7,
                        help="Minimum detection confidence for Mediapipe")
    parser.add_argument("--mptc", type=float, default=0.5,
                        help="Minimum tracking confidence for Mediapipe")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Prediction confidence threshold")
    parser.add_argument("--rate", type=float, default=0.001,
                        help="Learning rate of the model loaded")
    return parser.parse_args()


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

def build_model(num_classes):
    """Creates the per-frame Dense model architecture for alphabet recognition."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((42, )),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])

    return model

def draw_info(image, fps, word):
    cv2.putText(image, "FPS: " + str(fps), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(image, word, (100, 300),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return image

def string_words(letter):
    global word, consecutive_frames, prev_sign
    if letter == prev_sign:
        consecutive_frames += 1
    else:
        consecutive_frames = 0
    if consecutive_frames >= 25:
        word += letter
        consecutive_frames = 0
    prev_sign = letter
    print("Detected letter:", letter)
    print("Accumulated word:", word)
    print("Consecutive frames:", consecutive_frames)


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


def main():
    args = get_args()

    # Get label names from dataset folder (each subdirectory is a label), sorted alphabetically
    actions = sorted([
        folder for folder in os.listdir(args.dataset)
        if os.path.isdir(os.path.join(args.dataset, folder))
    ])
    num_classes = len(actions)
    print("Alphabet labels:", actions)

    optimizer = extract_optimizer_from_path(args.weights, args.rate)

    # Build and load the model
    model = build_model(num_classes=num_classes)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.load_weights(args.weights)
    print("Loaded model weights from:", args.weights)

    # Initialize Mediapipe Holistic
    mp_holistic = mp.solutions.holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        min_detection_confidence=args.mpdc,
        min_tracking_confidence=args.mptc
    )

    # FPS measurement utility
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Open the webcam
    camera_index = find_first_available_camera()
    if camera_index is None:
        print("No camera available.")
        return
    cap = cv2.VideoCapture(camera_index)

    with mp_holistic.Holistic(min_detection_confidence=args.mpdc,
                              min_tracking_confidence=args.mptc) as holistic:
        while cap.isOpened():

            fps = cvFpsCalc.get()
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            output_frame = frame.copy()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            hand_landmarks = extract_keypoints(results)

            if np.any(hand_landmarks):
                keypoints = calc_landmark_list(image, hand_landmarks)
                processed_keypoints = pre_process_landmark(keypoints)

                # Perform prediction (per-frame classification)
                pred_probs = model.predict(np.array([processed_keypoints]))[0]
                print("Predicted probabilities:", pred_probs)
                pred_idx = np.argmax(pred_probs)
                predicted_letter = actions[pred_idx]
                confidence = pred_probs[pred_idx]

                # Update accumulated word if prediction is stable
                string_words(predicted_letter)

                # Overlay prediction and confidence on the frame
                cv2.putText(image, "Action: " + predicted_letter, (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(image, "Confidence: " + str(round(confidence, 2)), (30, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            image = draw_info(image, fps, word)

            cv2.imshow("Alphabet Recognition", image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
