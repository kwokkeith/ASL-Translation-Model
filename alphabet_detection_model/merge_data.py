import argparse
import os
import shutil
import tempfile


def get_next_sequence_number(action_path):
    """Finds the next available sequence number for an action."""
    existing_sequences = sorted([
        int(folder) for folder in os.listdir(action_path) if folder.isdigit()
    ])
    return max(existing_sequences, default=-1) + 1


def merge_datasets(input_dirs, output_dir):
    """Merges multiple `mp_data` folders into one, \
    ensuring unique sequence numbers."""

    # Check if output directory is one of the input directories
    is_circular = output_dir in input_dirs

    # if circular, write to temp file
    temp_dir = None
    if is_circular:
        temp_dir = tempfile.mkdtemp()
        print(
            f"Circular in detected. Writing to temporary directory {temp_dir}")
        final_output_dir = temp_dir
    else:
        final_output_dir = output_dir

    # Ensure output directory exists
    os.makedirs(final_output_dir, exist_ok=True)

    for input_dir in input_dirs:
        input_dir = input_dir.strip()
        # Check if input_dir exist otherwise ignore
        if not os.path.exists(input_dir):
            print(f"Skipping missing dataset: {input_dir}")
            continue

        print(f"Merging dataset: {input_dir} -> {final_output_dir}")

        for action in os.listdir(input_dir):
            action_source_path = os.path.join(input_dir, action)
            action_dest_path = os.path.join(final_output_dir, action)

            if not os.path.isdir(action_source_path):
                continue  # Skip non-folder files

            # Ensure action folder exists
            os.makedirs(action_dest_path, exist_ok=True)

            for sequence in sorted(os.listdir(action_source_path)):
                sequence_source_path = os.path.join(
                    action_source_path, sequence)
                if not os.path.isdir(sequence_source_path):
                    continue  # Skip non-folder files

                # Find the next sequence number in the destination folder
                new_sequence_num = get_next_sequence_number(action_dest_path)
                new_sequence_path = os.path.join(
                    action_dest_path, str(new_sequence_num))

                shutil.copytree(sequence_source_path, new_sequence_path)
                print(
                    f"Copied {sequence_source_path} -> {new_sequence_path}")

    if is_circular:
        print(f"Replacing {output_dir} with merged dataset from {temp_dir}.")
        shutil.rmtree(output_dir)  # Remove old dataset
        # Move merged data to original location
        shutil.move(temp_dir, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description="Merges multiple action datasets together.")
    parser.add_argument("--i", type=str, required=True, nargs='+',
                        help="List of input datasets \
                        (i.e. mp_data1/, mp_data2/)")
    parser.add_argument("--o", type=str, required=True,
                        help="Output path to the resultant dataset")
    args = parser.parse_args()

    merge_datasets(args.i, args.o)


if __name__ == "__main__":
    main()
