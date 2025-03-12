import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def draw_styled_landmarks(image, results):
    """Draws styled landmarks for pose, hands, and face using MediaPipe."""
    # Draw Pose landmarks
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
    )

    # Draw Left Hand landmarks
    mp_drawing.draw_landmarks(
        image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
    )

    # Draw Right Hand landmarks
    mp_drawing.draw_landmarks(
        image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
    )

    # Draw Face landmarks
    mp_drawing.draw_landmarks(
        image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
    )

def extract_keypoints(results):
    """Extracts keypoints from the MediaPipe Holistic model."""
    pose = np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                    ) if results.pose_landmarks else np.zeros((33, 4))

    lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
                  ) if results.left_hand_landmarks else np.zeros((21, 3))

    rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
                  ) if results.right_hand_landmarks else np.zeros((21, 3))

    face = np.array([[lm.x, lm.y, lm.z] for lm in results.face_landmarks.landmark]
                    ) if results.face_landmarks else np.zeros((468, 3))

    return np.concatenate([pose.flatten(), lh.flatten(), rh.flatten(), face.flatten()])
