import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse


def create_label_features(data_path, skip_frame):
    """Preprocesses the data and creates the necessary features and labels"""
    X = []  # List to store frame data
    Y = []  # List to store labels

    # Optionally, create a mapping from label names to numeric indices
    labels = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    # Loop over each label directory
    for label in labels:
        label_path = os.path.join(data_path, label)
        # Loop over each sequence directory under the label
        for sequence in sorted(os.listdir(label_path)):
            sequence_path = os.path.join(label_path, sequence)
            if not os.path.isdir(sequence_path):
                continue
            # Get sorted list of frame files in the sequence
            frame_files = sorted(os.listdir(sequence_path))
            # Only load every skip_frame-th frame
            for i, frame_file in enumerate(frame_files):
                if not frame_file.endswith('.npy'):
                    continue
                if skip_frame != 0:
                    if i % skip_frame != 0:
                        continue
                frame_path = os.path.join(sequence_path, frame_file)
                frame = np.load(frame_path)  # Load the .npy file
                X.append(frame)
                Y.append(label_to_index[label])

    # Convert lists to numpy arrays if needed
    X = np.array(X)
    Y = np.array(Y)
    print("Total frames loaded:", len(X))
    print("Label mapping:", label_to_index)
    return X, Y


def get_storage_directory(base_dir, skip, testsize):
    """Generates a new unique folder name for storing dataset splits"""
    config_prefix = f"skip_{skip}_testsize_{testsize}"
    existing_folders = [folder for folder in os.listdir(
        base_dir) if folder.startswith(config_prefix)]

    # Find the next available index
    # If folders exist, create the next numbered version
    next_index = len(existing_folders)

    # Create the full directory path
    save_dir = os.path.join(base_dir, f"{config_prefix}_{next_index}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip", type=int, default="0",
                        help="Skip frames in attempt to generalise model")
    parser.add_argument("--sl", type=int, default="60",
                        help="Number of frames in sequence data collected.")
    parser.add_argument("--testsize", type=int, default="5",
                        help="The test size percentage to split the data")
    parser.add_argument("--i", type=str, default="mp_data",
                        help="Test dataset (i.e. mp_data) \
                        to use for splitting")
    args = parser.parse_args()

    # path for exported data, numpy arrays
    data_path = os.path.join(args.i)
    processed_data_path = os.path.join(args.i + '_processed')

    # Ensure processed data path exist
    os.makedirs(processed_data_path, exist_ok=True)

    # Scan the data_path directory and retrieve folder names to get actions
    if os.path.exists(data_path):
        actions = np.array([folder for folder in os.listdir(
            data_path) if os.path.isdir(os.path.join(data_path, folder))])
    else:
        print(f"Could not find the directory {data_path}")
        return

    feature, target = create_label_features(data_path, args.skip)

    X_train, X_test, Y_train, Y_test = train_test_split(
        feature, target, test_size=args.testsize/100.0)

    # Get a logical directory to save the dataset
    save_dir = get_storage_directory(
        processed_data_path, args.skip, args.testsize)

    # Save datasets
    np.save(os.path.join(save_dir, "X_train.npy"), X_train)
    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "Y_train.npy"), Y_train)
    np.save(os.path.join(save_dir, "Y_test.npy"), Y_test)

    print(f"Data saved to {save_dir}")


if __name__ == "__main__":
    main()
