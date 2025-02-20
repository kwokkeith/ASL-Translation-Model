import cv2
import numpy as np
import os
import mediapipe as mp
import argparse
from utils import draw_styled_landmarks, extract_keypoints, \
    mediapipe_detection, find_first_available_camera
import shutil

mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing Utilities


def cleanup_empty_folders(base_path):
    """Removes empty folders inside the given base path."""
    # Reverse order search for any empty folders
    for root, dirs, _ in os.walk(base_path, topdown=False):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            # Empty Folders
            if os.path.isdir(folder_path) and not os.listdir(folder_path):
                os.rmdir(folder_path)
                print(f"Removed empty folder: {folder_path}")


def remove_last_saved_sequence(data_path, actions, start_sequence):
    """Removes the first non-empty sequence when interrupted, \
    searching in reverse order."""
    for action in actions:
        action_path = os.path.join(data_path, action)

        # Get all sequence folders (sorted in reverse order)
        sequence_folders = sorted([
            int(folder) for folder in os.listdir(action_path)
            if folder.isdigit()
        ], reverse=True)  # Reverse sorting for backward search

        for last_sequence in sequence_folders:
            last_sequence_path = os.path.join(action_path, str(last_sequence))

            # Check if the folder contains data (not empty)
            if len(os.listdir(last_sequence_path)) > 0:
                if last_sequence < start_sequence:
                    return
                else:
                    print("Removing first non-empty sequence " +
                          "(backward search): " +
                          f"{last_sequence_path}")
                # Remove the first found non-empty sequence
                shutil.rmtree(last_sequence_path)
                return  # Exit after deleting the first non-empty sequence


def setup_folders(data_path, actions, no_sequences):
    """Cleans up empty folders and sets up directories for data collection."""
    os.makedirs(data_path, exist_ok=True)  # Ensure base directory exists

    for action in actions:
        action_path = os.path.join(data_path, action)
        os.makedirs(action_path, exist_ok=True)

        # Cleanup empty sequence folders before continuing
        cleanup_empty_folders(action_path)

        # Get the highest sequence number
        existing_sequences = [
            int(folder) for folder in os.listdir(action_path)
            if folder.isdigit()
        ]

        # Start numbering from the next available sequence number
        start_sequence = max(existing_sequences, default=-1) + 1
        print(f"Using start sequence: {start_sequence}")

        # Create new sequence folders
        for sequence in range(start_sequence, start_sequence + no_sequences):
            sequence_path = os.path.join(action_path, str(sequence))
            os.makedirs(sequence_path, exist_ok=True)

    return start_sequence


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions",
                        type=str, required=True, nargs='+',
                        help="actions to collect data for")
    parser.add_argument("--ns", type=int, default=30,
                        help="number of sequences to collect")
    parser.add_argument("--sl", type=int, default=60,
                        help="number of frames to collect per sequence")
    parser.add_argument("--mpdc", type=float, default=0.5,
                        help="Minimum detection confidence mediapipe model")
    parser.add_argument("--mptc", type=float, default=0.5,
                        help="Minimum tracking confidence mediapipe model")
    parser.add_argument("--wait", type=int, default=2000,
                        help="Wait time (ms) in between collection frames")
    args = parser.parse_args()
    print(f"actions to record {args.actions}")

    # path for exported data, numpy arrays
    data_path = os.path.join('mp_data')

    # actions to detect
    actions = np.array(args.actions)

    # number of videos worth of data
    no_sequences = args.ns

    # if videos are going to be 30 frames in length
    # 30 frames would be used to detect the action
    # 30 * 1662 keypoints data to detect the action
    sequence_length = args.sl

    # Activates video camera and captures sequences of actions
    # Find the first available camera
    camera_index = find_first_available_camera()

    if camera_index is None:
        print("Unable to find a camera that is available")
        return

    cap = cv2.VideoCapture(camera_index)

    # Only setup collection folder if we can open the videocapture
    # Setup collection folders
    start_sequence = setup_folders(data_path, actions, no_sequences)

    try:
        # Set mediapipe model
        with mp_holistic.Holistic(min_detection_confidence=args.mpdc,
                                  min_tracking_confidence=args.mptc) \
                as holistic:
            # Loop through actions
            for action in actions:
                # Loop through sequences aka videos
                for sequence in range(no_sequences):
                    # Loop through video length aka sequence length
                    actual_sequence = start_sequence + sequence
                    for frame_num in range(sequence_length):

                        # Read feed
                        ret, frame = cap.read()

                        # Detections to get landmarks face, shoulder and hands
                        image, results = mediapipe_detection(frame, holistic)

                        # Draw landmarks
                        draw_styled_landmarks(image, results)

                        # Apply wait logic
                        # so that we can take a break before recollecting data
                        # Display text to give instructions and state
                        if frame_num == 0:
                            cv2.putText(image, f"Collecting {action}...",
                                        (120, 250), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 4, cv2.LINE_AA)
                            cv2.putText(image,
                                        f"Collecting frames for {action}",
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3, cv2.LINE_AA)
                            cv2.putText(image,
                                        f"Video Number {sequence+1} of " +
                                        f"{no_sequences}",
                                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3, cv2.LINE_AA)
                            cv2.putText(image,
                                        "Actual Video Number: " +
                                        f"{actual_sequence}",
                                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)
                            cv2.waitKey(args.wait)
                        else:
                            cv2.putText(image,
                                        f"Collecting frames for {action}",
                                        (50, 100), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3, cv2.LINE_AA)
                            cv2.putText(image,
                                        f"Video Number {sequence+1} of " +
                                        f"{no_sequences}",
                                        (50, 150), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3, cv2.LINE_AA)
                            cv2.putText(image,
                                        "Actual Video Number: " +
                                        f"{actual_sequence}",
                                        (50, 200), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 3, cv2.LINE_AA)
                            # Show to screen
                            cv2.imshow('OpenCV Feed', image)

                        # Export keypoints
                        keypoints = extract_keypoints(results)
                        npy_path = os.path.join(
                            data_path, action,
                            str(actual_sequence),
                            str(frame_num))
                        np.save(npy_path, keypoints)

                        # Break gracefully
                        if cv2.waitKey(10) & 0xFF == ord('q'):
                            raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("\n Data collection interrupted. Saving collected data...")
        remove_last_saved_sequence(data_path, actions, start_sequence)
        cleanup_empty_folders(data_path)

    finally:
        # Release captures
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
