import numpy as np
import argparse
import os

def augment_keypoints(keypoints):
    """Applies small transformations to MediaPipe keypoints.
    Fast, keeps realistic pose structures.
    But... may not capture new unseen poses"""

    # 1. Small random scaling
    scale_factor = np.random.uniform(0.95, 1.05)
    keypoints_scaled = keypoints * scale_factor

    # 2. Small random translation (shift)
    translation = np.random.uniform(-0.05, 0.05, keypoints.shape)
    keypoints_translated = keypoints_scaled + translation

    # 3. Small rotation in 2D (X, Y only)
    theta = np.radians(np.random.uniform(-5, 5))  # Random angle
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta), np.cos(theta)]])
    
    keypoints_rotated = keypoints_translated.copy()
    # keypoints_rotated[:, :2] = np.dot(keypoints_rotated[:, :2], rotation_matrix)  # Rotate x, y only
    if keypoints_rotated.shape[1] >= 2:  # Ensure at least (x, y) exist
        keypoints_rotated[:, :2] = np.dot(keypoints_rotated[:, :2], rotation_matrix)


    return keypoints_rotated


def add_jitter(sequence, noise_level=0.01):
    """Adds small random noise to each frame."""
    noise_level = 0
    return sequence + np.random.normal(0, noise_level, sequence.shape)

def process_dataset(input_folder, output_folder, loop):
    """Processes input dataset, saves original and augmented keypoints while
    maintaining order and adding new sequences starting from the highest sequence number + 1."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for action in sorted(os.listdir(input_folder)):
        if action.startswith("."):  # Ignore hidden files
            continue

        print(f"Saving for {action}")
        action_path = os.path.join(input_folder, action)
        if not os.path.isdir(action_path):
            continue

        output_action_path = os.path.join(output_folder, action)
        os.makedirs(output_action_path, exist_ok=True)

        # Get the highest existing sequence index from the input dataset
        existing_sequences = sorted([
            int(seq) for seq in os.listdir(action_path) if seq.isdigit()
        ])

        if not existing_sequences:
            print(f"No sequences found for action '{action}'. Skipping...")
            continue

        highest_sequence_num = existing_sequences[-1]  
        next_sequence_num = highest_sequence_num + 1  

        for sequence in existing_sequences:
            sequence_path = os.path.join(action_path, str(sequence))
            output_sequence_path = os.path.join(output_action_path, str(sequence))
            os.makedirs(output_sequence_path, exist_ok=True)

            for frame_file in sorted(
                [f for f in os.listdir(sequence_path) if f.endswith('.npy')],
                key=lambda x: int(''.join(filter(str.isdigit, x.split('.')[0]))) if x.split('.')[0].isdigit() else float('inf')
            ):
                frame_path = os.path.join(sequence_path, frame_file)

                output_frame_path = os.path.join(output_sequence_path, frame_file)

                if not os.path.exists(frame_path):
                    print(f"Warning: Skipping missing file {frame_path}")
                    continue

                try:
                    keypoints = np.load(frame_path)
                except Exception as e:
                    print(f"Error loading {frame_path}: {e}")
                    continue  

                # Ensure correct shape before saving
                if keypoints.ndim == 1:
                    expected_size = 1662  
                    if keypoints.size != expected_size:
                        raise ValueError(f"Unexpected keypoints size: {keypoints.size}, expected {expected_size}")
                elif keypoints.ndim == 2 and keypoints.shape[1] == 3:
                    keypoints = keypoints.flatten()  

                # Save the original keypoints
                np.save(output_frame_path, keypoints)

                keypoints_reshaped = keypoints.reshape(-1, 3)

                # Generate `loop` number of augmented sequences
                for i in range(loop):
                    augmented_sequence_num = next_sequence_num + i
                    augmented_sequence_path = os.path.join(output_action_path, str(augmented_sequence_num))
                    os.makedirs(augmented_sequence_path, exist_ok=True)  # Ensure directory exists

                    output_augmented_frame_path = os.path.join(augmented_sequence_path, frame_file)

                    # Apply augmentations
                    keypoints_augmented = augment_keypoints(keypoints_reshaped)
                    keypoints_jittered = add_jitter(keypoints_augmented)

                    # Save the augmented keypoints
                    print(f"saving frame from {frame_path} --> {output_augmented_frame_path}")
                    np.save(output_augmented_frame_path, keypoints_jittered.flatten())

            next_sequence_num += loop  # Increment to prevent overwriting

    print(f"Original and augmented dataset saved to: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Augment MediaPipe keypoints dataset.")
    parser.add_argument("--i", type=str, required=True, help="Path to the input dataset folder")
    parser.add_argument("--o", type=str, required=True, help="Path to save the augmented dataset")
    parser.add_argument("--n", type=int, default=1, help="Number of loops to generated keypoints")

    args = parser.parse_args()
    
    process_dataset(args.i, args.o, args.n)