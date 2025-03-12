import numpy as np
import os
import argparse
from scipy.fft import fft, fftfreq


# load keypoints for sequence_length number of frames in a folder (e.g mp_data_dynamic_hands_only/hello/0), used in compute_ftt_features function.
def load_keypoints(base_path, action, sequence, sequence_length):
    keypoints = []
    sequence_path = os.path.join(base_path, action, str(sequence))
    
    if not os.path.exists(sequence_path):
        return None

    for frame_num in range(sequence_length):
        file_path = os.path.join(sequence_path, f"{frame_num}.npy")
        if os.path.exists(file_path):
            keypoints.append(np.load(file_path))
    
    return np.array(keypoints) if keypoints else None

# Function to compute FFT features
def compute_fft_features(motion_series, sequence_length, fps):
    fft_values = np.abs(fft(motion_series))[:sequence_length // 2]
    freqs = fftfreq(sequence_length, 1 / fps)[:sequence_length // 2]
    
    mean_freq = np.sum(freqs * fft_values) / np.sum(fft_values)  # Weighted mean frequency
    dominant_freq = freqs[np.argmax(fft_values)]  # Peak frequency
    
    return mean_freq, dominant_freq

parser = argparse.ArgumentParser(description="We input static & dynamic data path (either train or test data) into this model, to output ftt features & labels .npy files, which we can use to train/visualize our model.")

parser.add_argument("--dataset_dynamic_path", type=str, required=True, help="Path to the dynamic gestures dataset")
parser.add_argument("--dataset_static_path", type=str, help="Path to the static gestures dataset")
parser.add_argument("--sequence_length", type=int, default=30, help="Number of frames per sequence")
parser.add_argument("--fps", type=int, default=15, help="Frame rate of sequences")
parser.add_argument("--mode", type=str, choices=["train", "test"], required=True, help="Mode: train or test")
parser.add_argument("--output_features", type=str, required=True, help="Filename for saving FFT features")
parser.add_argument("--output_labels", type=str, required=True, help="Filename for saving labels")
args = parser.parse_args()

# Define datasets from arguments
datasets = {
    args.dataset_dynamic_path: 0,  # Dynamic Gestures (label 0)
    args.dataset_static_path: 1    # Static Gestures (label 1)
}


os.makedirs("fft_features",exist_ok=True)
os.makedirs("labels",exist_ok=True)
# Array to hold all motion magnitudes across datasets for global normalization
all_motion_magnitudes = []


# for each keypoint data, calculate movements, then calculate global mean and std, which will be used in the later for loop
for dataset, label in datasets.items():
    base_path = dataset
    actions = os.listdir(base_path)

    for action in actions:
        action_path = os.path.join(base_path, action)
        if not os.path.isdir(action_path):
            continue
        
        for sequence in os.listdir(action_path):
            sequence_path = os.path.join(action_path, sequence)
            if not os.path.isdir(sequence_path):
                continue
            
            keypoint_data = load_keypoints(base_path, action, sequence, args.sequence_length)
            if keypoint_data is not None:
                motion_magnitudes = [np.linalg.norm(keypoint_data[i] - keypoint_data[i - 1])
                                     for i in range(1, len(keypoint_data))]
                all_motion_magnitudes.extend(motion_magnitudes)


if len(all_motion_magnitudes) > 0:
    global_mean = np.mean(all_motion_magnitudes)
    global_std = np.std(all_motion_magnitudes)
else:
    global_mean, global_std = 0, 1  # Avoid division by zero

print(f"Global Motion Magnitude Mean: {global_mean:.4f}, Std: {global_std:.4f}")

X = []  # FFT feature vectors
y = []  # Labels (0 = Dynamic, 1 = Static)
metadata = []  # Stores (action, sequence) pairs for data analysis


# we then perform the same calulations, this time normalizing with the global mean and std, then saving in numpy array
for dataset, label in datasets.items():
    base_path = dataset
    actions = os.listdir(base_path)

    for action in actions:
        action_path = os.path.join(base_path, action)
        if not os.path.isdir(action_path):
            continue
        
        for sequence in os.listdir(action_path):
            sequence_path = os.path.join(action_path, sequence)
            if not os.path.isdir(sequence_path):
                continue
            
            keypoint_data = load_keypoints(base_path, action, sequence, args.sequence_length)
            if keypoint_data is not None:
                motion_magnitudes = [np.linalg.norm(keypoint_data[i] - keypoint_data[i - 1])
                                     for i in range(1, len(keypoint_data))]
                motion_magnitudes = np.array(motion_magnitudes)
                motion_magnitudes = (motion_magnitudes - global_mean) / global_std  # Normalize
                
                # Compute FFT features
                mean_freq, dominant_freq = compute_fft_features(motion_magnitudes, args.sequence_length, args.fps)
                
                # Store FFT features, labels, and metadata
                X.append([mean_freq, dominant_freq])
                y.append(label)
                if args.mode == "train":
                    metadata.append((action, sequence))
                

# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)

fft_features_file = os.path.join("fft_features", args.output_features)
labels_file = os.path.join("labels", args.output_labels)
# Save results
np.save(fft_features_file, X)
np.save(labels_file, y)
if args.mode == "train":
    np.save("metadata.npy", np.array(metadata, dtype=object))
    print(f"Metadata saved as metadata.npy")

print(f"Processed {len(X)} sequences. FFT Features & Labels Saved!")
