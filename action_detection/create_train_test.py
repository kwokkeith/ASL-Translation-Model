import numpy as np
import os
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import argparse


def create_label_features(actions,
                          sequence_length, data_path, skip_frame):
    """Preprocesses the data and creates the necessary features and labels"""
    # Create labels for action for which model has to predict
    label_map = {label: num for num, label in enumerate(actions)}
    print(f"Label map:\n{label_map}")
    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(data_path, action)

        # Dynamically get all sequence folders (integer-named folders)
        sequence_folders = [int(folder) for folder in os.listdir(
            action_path) if folder.isdigit()]
        sequence_folders.sort()  # Ensure they are in numerical order

        for sequence in sequence_folders:
            window = []
            sequence_folder_path = os.path.join(action_path, str(sequence))

            # Process frames
            for frame_num in range(0, sequence_length, skip_frame + 1):
                frame_path = os.path.join(
                    sequence_folder_path, f"{frame_num}.npy")

                if os.path.exists(frame_path):
                    res = np.load(frame_path)
                    window.append(res)
                else:
                    print(f"Warning: Missing frame {frame_path}")

            # Append processed sequence
            sequences.append(window)
            labels.append(label_map[action])  # Target variable

    X = np.array(sequences)  # Form feature set
    # One-hot encoding for labels to be predicted
    Y = to_categorical(labels).astype(int)
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

    feature, target = create_label_features(
        actions, args.sl, data_path, args.skip)

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
