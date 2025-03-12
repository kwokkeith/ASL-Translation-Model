import os
import numpy as np
import argparse
from pathlib import Path
import mediapipe as mp
from collections import defaultdict
import shutil

#input is a dataset folder (e.g /mp_data/) containing frames data (1663)
#output is a dataset folder (e.g /mp_data_hands_only/) containing only hands data in frames (126)
mp_holistic = mp.solutions.holistic  # Needed to match the original keypoints
HAND_LANDMARKS_COUNT = 21  # Each hand has 21 keypoints
 
def detect_hand_start(full_keypoints):
    """Detects the starting index of hand keypoints dynamically."""
    total_keypoints = len(full_keypoints)
    left_hand_start = 1536  
    right_hand_start = left_hand_start + 63  

    if total_keypoints != 1662:
        print(f"Expected 1662 keypoints, but found {total_keypoints}.")
    
    return left_hand_start, right_hand_start

def extract_hand_keypoints(full_keypoints):
    """Extracts only the hand keypoints dynamically (left & right) from a 1662-keypoint array."""
    left_hand_start, right_hand_start = detect_hand_start(full_keypoints)
    left_hand = full_keypoints[left_hand_start:left_hand_start + 63]  
    right_hand = full_keypoints[right_hand_start:right_hand_start + 63]

    if len(left_hand) != 63:
        left_hand = np.zeros(63)
    if len(right_hand) != 63:
        right_hand = np.zeros(63)

    return np.concatenate([left_hand, right_hand])  

def process_existing_data(input_folder, output_folder):
    """Processes all existing .npy files and extracts only hand keypoints."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    os.makedirs(output_folder, exist_ok=True)  

    for action_folder in input_folder.iterdir():
        if not action_folder.is_dir():
            continue  # Skip non-folder items

        action_output_path = output_folder / action_folder.name
        os.makedirs(action_output_path, exist_ok=True)

        for sequence_folder in action_folder.iterdir():
            if not sequence_folder.is_dir():
                continue

            sequence_output_path = action_output_path / sequence_folder.name
            os.makedirs(sequence_output_path, exist_ok=True)

            for npy_file in sequence_folder.glob("*.npy"):
                full_keypoints = np.load(npy_file)

                # Extract only hand keypoints
                hand_keypoints = extract_hand_keypoints(full_keypoints)

                # Save processed file
                np.save(sequence_output_path / npy_file.name, hand_keypoints)

    print("All hand keypoints extracted successfully!")

def remove_outlier_folders(base_folder, z_threshold=2.0):
    """Detects and removes entire sequence folders if they contain outliers."""
    base_folder = Path(base_folder)
    folder_outliers = defaultdict(set)  # Stores folders to be deleted

    for gesture_folder in base_folder.iterdir():
        if not gesture_folder.is_dir():
            continue

        file_means = {}
        sequence_to_files = defaultdict(list)

        for sequence_folder in gesture_folder.iterdir():
            if not sequence_folder.is_dir():
                continue

            for npy_file in sequence_folder.glob("*.npy"):
                data = np.load(npy_file)
                mean_value = np.mean(data)
                file_means[str(npy_file)] = mean_value  
                sequence_to_files[str(sequence_folder)].append(mean_value)  

        if not file_means:
            continue

        mean_values = np.array(list(file_means.values()))
        folder_mean = np.mean(mean_values)
        folder_std = np.std(mean_values)

        if folder_std == 0:
            print(f"{gesture_folder.name}: No variation in data. Skipping outlier detection.")
            continue

        for sequence_folder, mean_list in sequence_to_files.items():
            for mean_value in mean_list:
                z_score = (mean_value - folder_mean) / folder_std
                if abs(z_score) > z_threshold:
                    folder_outliers[gesture_folder.name].add(sequence_folder)
                    break  

    for gesture_name, outlier_folders in folder_outliers.items():
        for folder in outlier_folders:
            shutil.rmtree(folder)  
            print(f"Deleted Folder: {folder}")

    if not folder_outliers:
        print(" No significant outlier folders found!")
    else:
        print("Outlier folders deleted successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="This files serves to extract hand keypoints from holistic keypoints (pose, face, hand) and remove outliers in data, saving output in a folder we indicate in arguments(--output). we have two modes (--mode), test or train. test and train mode extracts the hand keypoints, while ONLY train modes removes the outliers ")
    parser.add_argument("--input", type=str, required=True, help="Path to input dataset folder.")
    parser.add_argument("--output", type=str, required=True, help="Path to output folder.")
    parser.add_argument("--mode", choices=["train", "test"], required=True, help="Processing mode: 'train' (removes outliers) or 'test' (keeps all data).")
    parser.add_argument("--z_threshold", type=float, default=2.0, help="Z-score threshold for outlier detection (only in 'train' mode).")
    
    args = parser.parse_args()

    print(f"Running in {args.mode.upper()} mode")
    print(f"Input Folder: {args.input}")
    print(f"Output Folder: {args.output}")
    
    process_existing_data(args.input, args.output)
    
    if args.mode == "train":
        remove_outlier_folders(args.output, args.z_threshold)
