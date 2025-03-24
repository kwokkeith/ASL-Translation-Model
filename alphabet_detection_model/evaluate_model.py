import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, TimeDistributed, BatchNormalization, Dropout
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
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

    print("X_test shape:", X_test.shape)
    print("Y_test shape:", Y_test.shape)   


    return X_test, Y_test

def build_model(num_classes):
    """Creates the per-frame inference model architecture."""
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

    return model

def print_confusion_matrix(y_true, y_pred, labels, save_dir, report=True):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save the confusion matrix as a heatmap    
    save_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"\nConfusion matrix heatmap saved to: {save_path}")
    plt.close()

    if report:
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, digits=4))



def evaluate_model(model, X_test, Y_test, labels, save_dir):
    """Evaluates the trained model and prints classification metrics."""
    print("\n*Running Evaluation on Test Data**")
    print("===================================")

    val_loss, val_acc = model.evaluate(X_test, Y_test, batch_size=128)

    print("\nAccuracy Score:", val_acc)
    print("\nLoss Score:", val_loss)
    
    Y_pred = model.predict(X_test)
    y_pred = np.argmax(Y_pred, axis=1)

    print_confusion_matrix(Y_test, y_pred, labels, save_dir)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model for per-frame inference")
    parser.add_argument("--model", type=str, required=True, help="Path to the saved model (.h5 file)")
    parser.add_argument("--ld", type=str, required=True, help="Path to the dataset folder containing data")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the dataset folder containing test data")
    parser.add_argument("--rate", type=float, default=0.001, help="Learning rate of the model to be loaded")

    args = parser.parse_args()

    # Load dataset
    X_test, Y_test = load_dataset(args.dataset)

    save_path = os.path.join("../" + args.dataset)

    # Load dataset to retrieve actions
    labels = sorted([
        folder for folder in os.listdir(args.ld)
        if os.path.isdir(os.path.join(args.ld, folder))
    ])
    print(labels)
    num_classes = len(labels)
    print(num_classes)

    # Extract number of classes from Y_test shape
    num_classes = len(np.unique(Y_test))

    # Build and compile the model
    model = load_model(args.model)
    # model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Load saved model weights
    # model.load_weights(args.weights, by_name=True, skip_mismatch=True)
    print(f"Loaded model from: {args.model}")

    # Evaluate model
    evaluate_model(model, X_test, Y_test, labels, args.dataset)

if __name__ == "__main__":
    main()
