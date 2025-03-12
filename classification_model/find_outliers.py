import numpy as np

# Load FFT features, labels, and metadata
X = np.load("fft_features.npy")  # FFT Features (samples, 2)
y = np.load("labels.npy")  # Labels (0 = Dynamic, 1 = Static)
metadata = np.load("metadata.npy", allow_pickle=True)  # Metadata: (action, sequence)

# Extract static gestures (indices in the full dataset)
static_indices = np.where(y == 1)[0]
static_mean_freqs = X[static_indices, 0]  # Mean Frequency
static_dominant_freqs = X[static_indices, 1]  # Dominant Frequency

# Define high-frequency threshold (e.g., above 1.5 Hz)
high_freq_threshold = 1.5

# Get indices of static gestures that exceed the threshold (in static subset)
high_freq_static_local_indices = np.where(static_dominant_freqs > high_freq_threshold)[0]

# Convert to original dataset indices
high_freq_static_indices = static_indices[high_freq_static_local_indices]

# Retrieve metadata (action, sequence) for high-frequency static gestures
high_freq_static_metadata = metadata[high_freq_static_indices]

# Print details
print(f"Total Static Gestures: {len(static_indices)}")
print(f"Static Gestures with High Frequency: {len(high_freq_static_indices)}\n")

print("High-Frequency Static Gestures (Action, Sequence):")
for i, idx in enumerate(high_freq_static_indices):
    action, sequence = metadata[idx]  # Get original metadata
    dominant_freq = X[idx, 1]  # Fetch from full dataset
    print(f"{i+1}. Action: {action}, Sequence: {sequence}, Dominant Frequency: {dominant_freq:.2f} Hz")
