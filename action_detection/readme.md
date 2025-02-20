# 📌 Action Recognition System

This repository contains Python scripts for training, evaluating, and running an LSTM-based action recognition system using **Mediapipe** and **TensorFlow**.

## 📂 Project Files

### **create\_train\_test.py**

📌 **Purpose:** This script processes collected action data and splits it into **training** and **testing** datasets.

🔹 **Arguments:**

- `--skip` (int, default=`0`): Number of frames to skip to generalize the model.
- `--sl` (int, default=`60`): Number of frames in a sequence.
- `--testsize` (int, default=`5`): The test dataset split percentage.

---

### **data\_collection.py**

📌 **Purpose:** Collects action sequences using **Mediapipe** for training an action recognition model.

🔹 **Arguments:**

- `--actions` (str, required): List of actions to collect data for.
- `--ns` (int, default=`30`): Number of sequences per action.
- `--sl` (int, default=`60`): Number of frames per sequence.
- `--mpdc` (float, default=`0.5`): Minimum Mediapipe detection confidence.
- `--mptc` (float, default=`0.5`): Minimum Mediapipe tracking confidence.
- `--wait` (int, default=`2000`): Wait time (ms) between data collection frames.

---

### **merge_datasets.py**

📌 **Purpose:** Merges multiple `mp_data` folders, ensuring unique sequence numbering, merging similar actions, and preventing circular merging issues.

🔹 **Arguments:**

- `--i` (str, required, nargs='+'): List of input datasets (e.g., `mp_data1/`, `mp_data2/`).
- `--o` (str, required): Output path to the resultant dataset.

---

### **evaluate\_model.py**

📌 **Purpose:** Loads a trained model, evaluates it on a test dataset, and reports accuracy and confusion matrices.

🔹 **Arguments:**

- `--weights` (str, required): Path to the saved model weights (`.h5`).
- `--dataset` (str, required): Path to the dataset folder for evaluation.
- `--rate` (float, default=`0.001`): Learning rate for loading the model.

---

### **train\_model.py**

📌 **Purpose:** Trains the LSTM model on a dataset and saves model weights.

🔹 **Arguments:**

- `--dataset` (str, required): Path to dataset (`mp_data_processed/...`).
- `--epochs` (int, default=`500`): Number of epochs for training.
- `--optimizer` (str, required, choices=[`Adam`, `SGD`, `AdamW`], default=`Adam`): Optimizer to use.
- `--rate` (float, required, default=`0.001`): Learning rate.

---

### **run\_live.py**

📌 **Purpose:** Runs real-time action recognition on a **live webcam feed** using a trained model.

🔹 **Arguments:**

- `--weights` (str, required): Path to the trained model weights (`.h5`).
- `--dataset` (str, required): Path to the dataset to get action names.
- `--threshold` (float, default=`0.5`): Minimum confidence threshold for predictions.
- `--rate` (float, default=`0.001`): Learning rate of the model.
- `--mpdc` (float, default=`0.5`): Minimum Mediapipe detection confidence.
- `--mptc` (float, default=`0.5`): Minimum Mediapipe tracking confidence.
- `--freeze` (int, default=`10`): Number of stable predictions required before displaying.
- `--sentences` (int, default=`5`): Number of actions to store and display on screen.

---

### **utils.py**

📌 **Purpose:** Utility functions for **Mediapipe processing, model building, and optimizer extraction**. Used in multiple scripts to reduce redundancy.

✅ **Includes:**

- `mediapipe_detection(image, model)`: Runs Mediapipe on an image.
- `draw_styled_landmarks(image, results)`: Draws pose, face, and hand landmarks.
- `extract_keypoints(results)`: Extracts keypoints from Mediapipe output.
- `build_model(input_shape, num_classes)`: Builds the LSTM model.
- `extract_optimizer_from_path(model_path)`: Extracts optimizer type from saved model name.

---

## 📦 **Installation & Setup**

1️⃣ Install dependencies:

```sh
pip install -r requirements.txt
```

2️⃣ Verify installation:

```sh
python -c "import tensorflow as tf; print(tf.__version__)"
python -c "import cv2; print(cv2.__version__)"
python -c "import mediapipe as mp; print(mp.__version__)"
python -c "import numpy as np; print(np.__version__)"
```

---

## 🚀 **Usage**

### **1️⃣ Collect Action Data**

```sh
python data_collection.py --actions jump run walk --ns 30 --sl 60
```

### **2️⃣ Create Train/Test Splits**

```sh
python create_train_test.py --skip 2 --sl 60 --testsize 5
```

### **3️⃣ Train the Model**

```sh
python train_model.py --dataset mp_data_processed/skip_2_testsize_5 --epochs 500 --optimizer Adam
```

### **4️⃣ Evaluate the Model**

```sh
python evaluate_model.py --weights Saved_Models/model_weights.h5 --dataset mp_data_processed/skip_2_testsize_5
```

### **5️⃣ Run Live Action Detection**

```sh
python run_live.py --weights Saved_Models/model_weights.h5 --dataset mp_data_processed/skip_2_testsize_5
```

---

## 📌 **Contributing**

Feel free to contribute by improving models, adding new action types, or enhancing the visualization! 🚀


