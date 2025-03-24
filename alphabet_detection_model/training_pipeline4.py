# This python script is used to train NN models with the relevant outputs required to analyse the model.
# This is done automatically to run each for each NN model and save its weights and an output csv

import utils
import os
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, SpatialDropout1D, LayerNormalization
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam, SGD, AdamW
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import json
from tensorflow.keras.regularizers import l2

labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 
          'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 
          'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

def load_dataset(dataset_dir, test_only=False):
    """Loads dataset. If test_only=True, loads only test data."""
    X_test_path = os.path.join(dataset_dir, "X_test.npy")
    Y_test_path = os.path.join(dataset_dir, "Y_test.npy")

    if test_only:
        if not all(os.path.exists(path) for path in [X_test_path, Y_test_path]):
            print(f"Error: Test files are missing in {dataset_dir}")
            exit(1)
        return np.load(X_test_path), np.load(Y_test_path)
    
    X_train_path = os.path.join(dataset_dir, "X_train.npy")
    Y_train_path = os.path.join(dataset_dir, "Y_train.npy")

    if not all(os.path.exists(path) for path in [X_train_path, X_test_path, Y_train_path, Y_test_path]):
        print(f"Error: One or more files are missing in {dataset_dir}")
        exit(1)

    return np.load(X_train_path), np.load(X_test_path), np.load(Y_train_path), np.load(Y_test_path)

def get_model_save_path(output_dir, dataset_name, optimizer, epochs, run):
    """Generates a unique folder to save model weights."""
    model_dir = os.path.join(output_dir, f"{dataset_name}_{optimizer}_epochs{epochs}_run{run}")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def build_model(input_shape, actions_count):
    """Builds and returns the LSTM model."""
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(64, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(32, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(16, activation='mish', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dense(actions_count, activation='softmax')
    ])

    return model

def evaluate_model(model, X_test, Y_test):
    """Evaluates model and returns performance metrics."""
    Y_pred = model.predict(X_test)
    Y_pred_classes = np.argmax(Y_pred, axis=1)

    if Y_test.ndim == 1:  
        Y_true_classes = Y_test
    else:  
        Y_true_classes = np.argmax(Y_test, axis=1)


    acc = accuracy_score(Y_true_classes, Y_pred_classes)
    f1 = f1_score(Y_true_classes, Y_pred_classes, average="weighted")
    conf_matrix = confusion_matrix(Y_true_classes, Y_pred_classes)
    class_report = classification_report(Y_true_classes, Y_pred_classes, output_dict=True)

    return acc, f1, conf_matrix, class_report

def train_and_evaluate(model, optimizer, X_train, Y_train, X_test, Y_test, X_ood_test, Y_ood_test, epochs, output_dir, dataset_name, run, batch_size, labels):
    """Trains the model, saves results, and evaluates performance."""
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    log_dir = os.path.join(output_dir, "Logs")
    os.makedirs(log_dir, exist_ok=True)
    tb_callback = TensorBoard(log_dir=log_dir)
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.000001)

    #early_stopping_callback = EarlyStopping(monitor='val_loss', patience=25, mode='min', restore_best_weights=True)

    print(f"Training model for {epochs} epochs...")
    if X_train.ndim==4:
        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)    

    history = model.fit(X_train, Y_train, epochs=epochs, shuffle=True, batch_size=batch_size, validation_split=0.2, callbacks=[lr_scheduler])
    #history = model.fit(X_train, Y_train, epochs=epochs, shuffle=True, validation_split=0.2, callbacks=[tb_callback, early_stopping_callback])

    # Save Model & Weights
    model_save_path = get_model_save_path(output_dir, dataset_name, optimizer.__class__.__name__, epochs, run)
    model.save(os.path.join(model_save_path, "model.h5"))
    model.save_weights(os.path.join(model_save_path, "model_weights.weights.h5"))

    # Plot Training Graphs
    plot_metric(history, 'loss', 'val_loss', "Loss vs Validation Loss", model_save_path, "Loss")
    plot_metric(history, 'accuracy', 'val_accuracy', "Accuracy vs Validation Accuracy", model_save_path, "Accuracy")

    # Evaluate In-Domain Data
    acc, f1, conf_matrix, class_report = evaluate_model(model, X_test, Y_test)
    save_confusion_matrix(conf_matrix, model_save_path, "Confusion_Matrix", labels)

    # Evaluate Out-of-Domain Data
    ood_acc, ood_f1, ood_conf_matrix, ood_class_report = evaluate_model(model, X_ood_test, Y_ood_test)
    save_confusion_matrix(ood_conf_matrix, model_save_path, "Confusion_Matrix_OOD", labels)

    # Save Training Results to CSV
    results_csv_path = os.path.join(output_dir, "training_results.csv")
    save_results_to_csv(results_csv_path, epochs, acc, f1, history.history["val_loss"][-1], history.history["val_accuracy"][-1], class_report, ood_class_report, conf_matrix, ood_conf_matrix)

    print(f"Model trained for {epochs} epochs and results saved successfully!")

def plot_metric(history, metric, val_metric, title, save_dir, filename):
    """Plots and saves training graphs."""
    plt.figure()
    plt.plot(history.history[metric], label=metric)
    plt.plot(history.history[val_metric], label=val_metric)
    plt.title(title)
    plt.xlabel('Epochs')
    plt.ylabel(metric)
    plt.legend()
    plt.savefig(os.path.join(save_dir, f"{filename}.png"))
    plt.close()

def save_confusion_matrix(conf_matrix, save_dir, filename, labels):
    """Saves the confusion matrix as a heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(filename.replace("_", " "))
    plt.savefig(os.path.join(save_dir, f"{filename}.png"))
    plt.close()

def save_results_to_csv(csv_path, epochs, acc, f1, loss, cat_acc, class_report, ood_class_report, conf_matrix, ood_conf_matrix):
    """Appends results to a CSV file."""
    df = pd.DataFrame({
        "Epochs": [epochs],
        "Accuracy": [acc],
        "F1-score": [f1],
        "Loss": [loss],
        "Categorical Accuracy": [cat_acc],
        "Precision": [class_report["weighted avg"]["precision"]],
        "Recall": [class_report["weighted avg"]["recall"]],
        "Conf Matrix": [conf_matrix],
        "Classification Report": [class_report],
        "Precision (OOD)": [ood_class_report["weighted avg"]["precision"]],
        "Recall (OOD)": [ood_class_report["weighted avg"]["recall"]],
        "Conf Matrix (OOD)": [ood_conf_matrix],
        "Classification Report (OOD)": [ood_class_report]
    })

    if os.path.exists(csv_path):
        df.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_path, mode='w', index=False)

def main():
    parser = argparse.ArgumentParser(description="Automated Training with Out-of-Domain Evaluation")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--ood_dataset", type=str, required=True, help="Path to out-of-domain dataset")
    parser.add_argument("--output", type=str, required=True, help="Directory to store models and logs")
    parser.add_argument("--start_epoch", type=int, default=100)
    parser.add_argument("--end_epoch", type=int, default=500)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--optimizer", type=str, default="Adam", choices=["Adam", "SGD", "AdamW"])
    parser.add_argument("--rate", type=float, default=0.001)
    parser.add_argument("--runs_per_epoch", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    X_train, X_test, Y_train, Y_test = load_dataset(args.dataset)
    X_ood_test, Y_ood_test = load_dataset(args.ood_dataset, test_only=True)
    number_of_classes = len(np.unique(Y_train))

    for epochs in range(args.start_epoch, args.end_epoch + 1, args.interval):
        for run in range(1, args.runs_per_epoch + 1):
            optimizer = {"Adam": Adam, "SGD": SGD, "AdamW": AdamW}[args.optimizer](learning_rate=args.rate, clipnorm=1.0)
            model = build_model(input_shape=(42,), actions_count=number_of_classes)
            train_and_evaluate(model, optimizer, X_train, Y_train, X_test, Y_test, X_ood_test, Y_ood_test, epochs, args.output, os.path.basename(args.dataset), run, args.batch_size, labels)

if __name__ == "__main__":
    main()
