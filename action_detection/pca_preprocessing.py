import os
import numpy as np
import argparse
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import joblib

def load_keypoints(data_path, actions, sequence_length, skip_frame):
    """Loads keypoints from dataset and applies PCA."""
    label_map = {label: num for num, label in enumerate(actions)}
    print(f"Label map:\n{label_map}")

    sequences, labels = [], []

    for action in actions:
        action_path = os.path.join(data_path, action)

        sequence_folders = [int(folder) for folder in os.listdir(action_path) if folder.isdigit()]
        sequence_folders.sort()  

        for sequence in sequence_folders:
            window = []
            sequence_folder_path = os.path.join(action_path, str(sequence))

            for frame_num in range(0, sequence_length, skip_frame + 1):
                frame_path = os.path.join(sequence_folder_path, f"{frame_num}.npy")

                if os.path.exists(frame_path):
                    keypoints = np.load(frame_path)
                    window.append(keypoints)
                else:
                    print(f"Warning: Missing frame {frame_path}")

            sequences.append(window)
            labels.append(label_map[action])  

    X = np.array(sequences)  
    Y = to_categorical(labels).astype(int)  
    return X, Y

def apply_pca(X, num_components=50, save_path=None):
    """Applies PCA to reduce dimensionality and saves the trained PCA model."""
    
    if X.ndim == 4:
        num_samples, num_timesteps, num_keypoints, num_features = X.shape
        X_reshaped = X.reshape(num_samples * num_timesteps, -1)  
    elif X.ndim == 3:
        num_samples, num_timesteps, num_features = X.shape
        X_reshaped = X.reshape(num_samples * num_timesteps, num_features)  
    else:
        raise ValueError(f"Unexpected input shape: {X.shape}")

    # Fit PCA
    pca = PCA(n_components=num_components)
    X_pca = pca.fit_transform(X_reshaped)

    explained_variance = sum(pca.explained_variance_ratio_) * 100
    print(f"PCA Reduced to {num_components} components, retaining {explained_variance:.2f}% variance.")

    # Save PCA model
    if save_path:
        joblib.dump(pca, save_path)
        print(f"PCA model saved at {save_path}")

    X_pca = X_pca.reshape(num_samples, num_timesteps, num_components)
    return X_pca, pca


def get_storage_directory(base_dir, skip, testsize, pca_components):
    """Generates a new unique folder name for storing PCA-processed dataset."""
    config_prefix = f"skip_{skip}_testsize_{testsize}_pca_{pca_components}"
    existing_folders = [folder for folder in os.listdir(base_dir) if folder.startswith(config_prefix)]
    next_index = len(existing_folders)

    save_dir = os.path.join(base_dir, f"{config_prefix}_{next_index}")
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def main():
    parser = argparse.ArgumentParser(description="PCA preprocessing for MediaPipe dataset")
    parser.add_argument("--skip", type=int, default=0, help="Skip frames for generalisation")
    parser.add_argument("--sl", type=int, default=60, help="Number of frames per sequence")
    parser.add_argument("--testsize", type=int, default=5, help="Percentage of data for testing")
    parser.add_argument("--pca", type=int, default=50, help="Number of PCA components")
    parser.add_argument("--i", type=str, default="mp_data", help="Dataset folder")
    args = parser.parse_args()

    data_path = os.path.join(args.i)
    processed_data_path = os.path.join(args.i + '_pca_processed')
    os.makedirs(processed_data_path, exist_ok=True)

    if os.path.exists(data_path):
        actions = np.array([folder for folder in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, folder))])
    else:
        print(f"Error: Dataset folder '{data_path}' not found.")
        return

    print("Loading dataset...")
    X, Y = load_keypoints(data_path, actions, args.sl, args.skip)

    print("Applying PCA...")

    # Create a subdirectory for the current PCA configuration
    save_dir = get_storage_directory(processed_data_path, args.skip, args.testsize, args.pca) 

    # Define path to save the PCA model
    pca_model_path = os.path.join(save_dir, "pca_model.pkl")

    # Apply PCA and save the model
    X_pca, pca = apply_pca(X, args.pca)
    joblib.dump(pca, pca_model_path)  # Save the trained PCA model

    # Ensure X_train and X_test have 3 dimensions (samples, timesteps, features)
    if X_pca.ndim == 4:
        X_pca = X_pca.reshape(X_pca.shape[0], X_pca.shape[1], -1)

    if args.testsize != 100.0:
        X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=args.testsize / 100.0, stratify=Y)

        np.save(os.path.join(save_dir, "X_train.npy"), X_train)
        np.save(os.path.join(save_dir, "Y_train.npy"), Y_train)
    else:
        X_test, Y_test = X_pca, Y

    np.save(os.path.join(save_dir, "X_test.npy"), X_test)
    np.save(os.path.join(save_dir, "Y_test.npy"), Y_test)

    print(f"PCA-processed data saved in: {save_dir}")

if __name__ == "__main__":
    main()
