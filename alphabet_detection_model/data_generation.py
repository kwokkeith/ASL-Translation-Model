import numpy as np
import argparse
import os

LEFT_TO_RIGHT_LANDMARKS = [
    0,  # Wrist remains the same
    1, 2, 3, 4,  # Thumb (same order)
    8, 7, 6, 5,  # Index finger (reverse order)
    12, 11, 10, 9,  # Middle finger (reverse order)
    16, 15, 14, 13,  # Ring finger (reverse order)
    20, 19, 18, 17,  # Pinky (reverse order)
]

# Paths to original (right-hand) and mirrored (left-hand) dataset
def process_dataset(input_folder, output_folder):
    """Mirrors keypoints from right-hand to left-hand using np.fliplr()."""

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for action in sorted(os.listdir(input_folder)):
        print(f"Processing {action}")
        action_path = os.path.join(input_folder, action)
        if not os.path.isdir(action_path):
            continue

        output_action_path = os.path.join(output_folder, action)
        os.makedirs(output_action_path, exist_ok=True)

        existing_sequences = sorted([
            int(seq) for seq in os.listdir(action_path) if seq.isdigit()
        ])

        if not existing_sequences:
            print(f"No sequences found for action '{action}'. Skipping...")
            continue

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

                # Load right-hand keypoints
                right_hand_keypoints = np.load(frame_path)
                left_hand_keypoints = right_hand_keypoints.reshape(-1, 2)
                left_hand_keypoints[:,0] *= -1
                left_hand_keypoints = left_hand_keypoints.flatten()

                # Save mirrored keypoints
                np.save(output_frame_path, left_hand_keypoints)

            print(f"Mirrored {sequence_path} → {output_sequence_path}")

    print(f"\n Left-hand dataset saved to: {output_folder}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mirror right-hand MediaPipe keypoints dataset to create left-hand keypoints using np.fliplr().")
    parser.add_argument("--i", type=str, required=True, help="Path to the right-hand dataset")
    parser.add_argument("--o", type=str, required=True, help="Path to save the left-hand dataset")
    
    args = parser.parse_args()
    
    process_dataset(args.i, args.o)
