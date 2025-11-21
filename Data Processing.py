import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path
from tqdm import tqdm
import pickle
import cv2
import numpy as np
import glob
import os # Import os for path joining and listdir (alternative to Path.iterdir)

# --- Configuration ---
model_path = 'MediaPipe Models/hand_landmarker.task'
train_data_dir = "Dataset/ASL Alphabet Train"
output_pickle_file = 'Misc Models/mediapipe_hand_world_landmarks.pickle'

# --- MediaPipe Setup ---
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# --- Data Collection ---

# Prepare a list of all class folders and initialize data structure
alphabet_folders = sorted([d for d in os.listdir(train_data_dir) if os.path.isdir(os.path.join(train_data_dir, d))])
final_landmarks = {alphabet: [] for alphabet in alphabet_folders}

print("Starting landmark detection and augmentation...")

# Loop through each alphabet folder, showing overall progress
for alphabet in tqdm(alphabet_folders, desc="Overall Progress"):
    alphabet_folder_path = os.path.join(train_data_dir, alphabet)

    # Use glob for efficient file path gathering
    image_paths = glob.glob(os.path.join(alphabet_folder_path, '*.jpg')) # Adjust extension if needed

    # Loop through all images in the current folder
    for image_path_str in image_paths:
        # Read image with OpenCV
        cv_image = cv2.imread(image_path_str)
        if cv_image is None:
            continue
            
        # Function to process an image and extract landmarks
        def process_image_and_extract(img):
            """Detects hand landmarks and returns them as a flattened list or None."""
            # Convert BGR to RGB and create MediaPipe Image
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            
            # Detect landmarks
            detection_result = detector.detect(image).hand_world_landmarks
            
            # Check if any hands were detected
            if detection_result:
                # Assuming single hand per image based on original loop logic: detection_result[0]
                # Flatten the list of WorldLandmark objects (x, y, z) into a single 1D list
                # Use NumPy for efficient extraction and flattening
                landmarks_np = np.array([[lm.x, lm.y, lm.z] for lm in detection_result[0]]).flatten()
                return landmarks_np.tolist() # Convert back to list for consistent structure
            return None

        # 1. Process original image
        original_landmarks = process_image_and_extract(cv_image)
        if original_landmarks:
            final_landmarks[alphabet].append(original_landmarks)

        # 2. Process flipped image (Augmentation)
        flipped_cv_image = cv2.flip(cv_image, 1)
        flipped_landmarks = process_image_and_extract(flipped_cv_image)
        if flipped_landmarks:
            final_landmarks[alphabet].append(flipped_landmarks)


# --- Save Data ---
print("\nSaving collected landmarks...")
with open(output_pickle_file, 'wb') as handle:
    pickle.dump(final_landmarks, handle)

# --- Final Report ---
total_landmarks = sum(len(lms) for lms in final_landmarks.values())
print(f"\n✅ Data collection complete. Total landmarks collected: **{total_landmarks}**")
print(f"Landmarks saved to: **{output_pickle_file}**")