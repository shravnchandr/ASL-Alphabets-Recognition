# 🖐️ Real-Time ASL Alphabet Classifier

## 💡 Overview

This project implements a real-time system for classifying **American Sign Language (ASL) alphabet gestures** using computer vision and machine learning. It leverages **Google's MediaPipe Hand Landmarker** for robust hand tracking and a lightweight **PyTorch** feedforward neural network for instant classification.

The system is designed for high-performance inference, capable of processing a live webcam feed and displaying the predicted ASL character directly on the video stream.

## 🛠️ Key Technologies

  * **Computer Vision/Tracking:** **MediaPipe Hand Landmarker** for extracting 3D world coordinates of 21 hand landmarks per hand.
  * **Machine Learning Framework:** **PyTorch** for defining, training, and running the classification model.
  * **Data Handling:** **NumPy** and **Scikit-learn** (`StandardScaler`) for feature processing and normalization.
  * **Interface:** **OpenCV (`cv2`)** for camera integration and real-time visualization.

## ⚙️ Project Structure and Pipeline

The project follows a three-stage pipeline:

1.  **Data Preparation (`collect_data.py`):**
      * Reads images from the training dataset.
      * Uses MediaPipe to extract **63 features** (21 landmarks $\times$ 3 world coordinates $x, y, z$) per hand.
      * Applies **horizontal flipping** as data augmentation.
      * Saves the raw landmarks to `mediapipe_hand_world_landmarks.pickle`.
2.  **Training and Export (`train_model.py`):**
      * Loads landmarks and labels.
      * Fits and saves a **`StandardScaler`** to normalize the 63 features.
      * Trains a small **3-layer PyTorch MLP**

[Image of a simple neural network diagram showing input, hidden layers, and output]
to classify the 28 ASL signs.
\* Saves the final model parameters as `hand_landmark_model_state.pth`.
3\.  **Inference (`real_time_inference.py`):**
\* Loads the trained **PyTorch model** and the saved **`StandardScaler`**.
\* Captures a live feed from the webcam.
\* In real-time, it detects hand landmarks, scales the features using the loaded `StandardScaler`, and feeds them to the PyTorch model for classification.
\* Displays the predicted ASL character (`A`, `B`, `space`, `del`, etc.) on the video frame.

-----

## 🚀 Setup and Installation

### Prerequisites

You need Python 3.8+ and the following libraries.

```bash
pip install torch torchvision torchaudio
pip install mediapipe opencv-python numpy scikit-learn tqdm
```

### Model and Data Files

1.  **MediaPipe Model:** Download the `hand_landmarker.task` model file and place it in the directory specified by `model_path` (e.g., `MediaPipe Models/`).
2.  **Dataset:** Ensure your ASL image dataset is structured correctly (e.g., `Dataset/ASL Alphabet Train/A/`, `Dataset/ASL Alphabet Train/B/`, etc.).

-----

## 💻 Usage

### Step 1: Data Collection

Run the script to extract landmarks from your dataset.

```bash
python collect_data.py
```

### Step 2: Training the Classifier

Train the model and save the `scaler.pkl` and `hand_landmark_model_state.pth` files.

```bash
python train_model.py
```

### Step 3: Real-Time Prediction

Run the inference script to test the classifier on your webcam. Press **'q'** to quit the application.

```bash
python real_time_inference.py
```

## 🐳 Docker Support

You can also run the Live Recognition application in a Docker container.

### Prerequisites (macOS)
On macOS, you need **XQuartz** to allow the container to display the GUI window.
1. Install XQuartz: `brew install --cask xquartz`
2. Open XQuartz, go to **Preferences > Security**, and check **"Allow connections from network clients"**.
3. Restart XQuartz (or log out and back in).
4. In your terminal, run: `xhost +localhost`

### Running with Docker Compose (Recommended)
Simply run:
```bash
docker-compose up
```

### Building Manually
```bash
docker build -t asl-recognition .
```
