import os
import argparse
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import multilabel_confusion_matrix, \
    accuracy_score, classification_report
from utils import extract_optimizer_from_path


def load_dataset(dataset_dir):
    """Loads the testing dataset from the specified directory."""
    X_test_path = os.path.join(dataset_dir, "X_test.npy")
    Y_test_path = os.path.join(dataset_dir, "Y_test.npy")

    if not all(os.path.exists(path) for path in [X_test_path, Y_test_path]):
        print(f"Error: Missing X_test.npy or Y_test.npy in {dataset_dir}")
        exit(1)

    X_test = np.load(X_test_path)
    Y_test = np.load(Y_test_path)

    return X_test, Y_test


def build_model(input_shape, num_classes):
    """Creates the LSTM model architecture."""
    model = Sequential([
        LSTM(64, return_sequences=True,
             activation='relu', input_shape=input_shape),
        LSTM(128, return_sequences=True, activation='relu'),
        LSTM(64, return_sequences=False, activation='relu'),
        Dense(64, activation='relu'),
        Dense(32, activation='relu'),
        Dense(num_classes, activation='softmax')  # Output layer
    ])
    return model


def evaluate_model(model, X_test, Y_test):
    """Evaluates the trained model and prints classification metrics."""
    print("\n*Running Evaluation on Test Data**")
    print("===================================")

    # Make predictions
    y_pred = model.predict(X_test)

    # Convert one-hot encoding to categorical labels
    y_true = np.argmax(Y_test, axis=1).tolist()
    y_pred = np.argmax(y_pred, axis=1).tolist()

    # Evaluation metrics
    print("\nConfusion Matrix:")
    print(multilabel_confusion_matrix(y_true, y_pred))
    print("\nAccuracy Score:", accuracy_score(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, digits=4))


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained LSTM model")
    parser.add_argument("--weights", type=str, required=True,
                        help="Path to the saved model weights (.h5 file)")
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to the dataset folder containing test data")
    parser.add_argument("--rate", type=float, default="0.001",
                        help="Learning rate of the model to be loaded")

    args = parser.parse_args()

    # Load dataset
    X_test, Y_test = load_dataset(args.dataset)

    # Extract number of classes from Y_test shape
    num_classes = Y_test.shape[1]  # based on one-hot encoding

    # Determine optimiser from model path
    optimizer = extract_optimizer_from_path(args.weights, args.rate)

    # Build and compile the model
    model = build_model(input_shape=(
        X_test.shape[1], X_test.shape[2]), num_classes=num_classes)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    # Load saved model weights
    model.load_weights(args.weights)
    print(f"Loaded weights from: {args.weights}")

    # Evaluate model
    evaluate_model(model, X_test, Y_test)


if __name__ == "__main__":
    main()
