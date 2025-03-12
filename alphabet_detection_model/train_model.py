import os
import utils
import argparse
import numpy as np
import tensorflow as tf
import csv
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, BatchNormalization, Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import AdamW, SGD, Adam
from evaluate_model import evaluate_model


def load_split_dataset(dataset_dir):
    """Loads the training and testing dataset from the specified directory."""
    X_train_path = os.path.join(dataset_dir, "X_train.npy")
    X_test_path = os.path.join(dataset_dir, "X_test.npy")
    Y_train_path = os.path.join(dataset_dir, "Y_train.npy")
    Y_test_path = os.path.join(dataset_dir, "Y_test.npy")

    if not all(os.path.exists(path) for path in [X_train_path, X_test_path, Y_train_path, Y_test_path]):
        print(f"Error: One or more files are missing in {dataset_dir}")
        exit(1)

    X_train = np.load(X_train_path)
    X_test = np.load(X_test_path)
    Y_train = np.load(Y_train_path)
    Y_test = np.load(Y_test_path)

    return X_train, X_test, Y_train, Y_test


def get_model_save_path(base_dir, dataset_name, optimizer, epochs):
    """Generates a unique folder name to save model weights"""
    model_dir = os.path.join(base_dir, f"{dataset_name}_{optimizer}_epochs{epochs}")
    existing_models = [folder for folder in os.listdir(base_dir) if folder.startswith(f"{dataset_name}_{optimizer}_epochs{epochs}")]
    next_index = len(existing_models)
    save_path = os.path.join(model_dir + f"_{next_index}")
    os.makedirs(save_path, exist_ok=True)
    return save_path


def main():
    parser = argparse.ArgumentParser(description="Train a model for per-frame action recognition")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset folder (e.g., mp_data_processed/skip_2_testsize_10_0)")
    parser.add_argument("--ld", type=str, required=True, help="Path to the dataset folder containing data")
    parser.add_argument("--epochs", type=int, default=500, help="Number of epochs for training")
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "AdamW"], help="Optimizer to use for the model")
    parser.add_argument("--rate", type=float, default=0.001, help="Learning rate for the model")
    parser.add_argument("--testsize", type=int, default="5", help="The test size percentage to split the data")

    args = parser.parse_args()
    dataset_path = args.dataset
    dataset_name = os.path.basename(dataset_path)

    X_train, X_test, Y_train, Y_test = load_split_dataset(dataset_path)
    num_classes = len(np.unique(Y_train))
    print(f"Number of classes: {num_classes}")
    # Load dataset to retrieve labels
    labels = sorted([
        folder for folder in os.listdir(args.ld)
        if os.path.isdir(os.path.join(args.ld, folder))
    ])
    
    # Setup TensorBoard
    log_dir = os.path.join("Logs")
    tb_callback = TensorBoard(log_dir=log_dir)
    # Callback for early stopping
    # es_callback = tf.keras.callbacks.EarlyStopping(patience=20, verbose=1)
    
    # Build Frame-Level Model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input((42, )),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(128, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(64, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])


    # Compile Model
    if args.optimizer == "Adam":
        model.compile(optimizer=Adam(learning_rate=args.rate), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    elif args.optimizer == "SGD":
        model.compile(optimizer=SGD(learning_rate=args.rate, momentum=0.9), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    elif args.optimizer == "AdamW":
        model.compile(optimizer=AdamW(learning_rate=args.rate, weight_decay=1e-4), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    print("Training model...")
    # Train Model
    try:
        model_training_history = model.fit(X_train, Y_train, 
                                           epochs=args.epochs, 
                                           batch_size=128,
                                           validation_split = args.testsize/100,
                                        #    shuffle=True,
                                           callbacks=[tb_callback])
    except KeyboardInterrupt:
        pass
    finally:
        # Print Model Summary
        print(model.summary())

        # Get save path for model weights
        os.makedirs("Saved_Models_New", exist_ok=True)
        model_save_path = get_model_save_path(
            "Saved_Models_New", dataset_name, args.optimizer, args.epochs)

        # Get plot of model_training_history
        # Visualise training and validation loss metrics
        plot_1 = utils.plot_metric(model_training_history,
                                   'loss',
                                   'val_loss',
                                   'Total Loss vs Total validation Loss')
        plot_1.savefig(os.path.join(model_save_path, "Loss"))
        plot_1.close()
        # visualise training and validation accuracy metrics
        plot_2 = utils.plot_metric(model_training_history,
                                   'accuracy',
                                   'val_accuracy',
                                   'Total Accuracy vs Total Validation Accuracy')
        plot_2.savefig(os.path.join(model_save_path, "Accuracy"))
        plot_2.close()

        # Save Model Weights
        weights_file = os.path.join(model_save_path, "model_weights.h5")
        model.save_weights(weights_file)

        print(f"Model weights saved at: {weights_file}")

        evaluate_model(model, X_test, Y_test, labels, model_save_path)

if __name__ == "__main__":
    main()
