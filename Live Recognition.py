import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import pickle
import torch
import torch.nn as nn
import time # Import for potential frame rate calculation

# --- Configuration Constants ---
MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
PREDICTION_TEXT_COLOR = (0, 255, 0)
PREDICTION_TEXT_POS = (10, 50)
MODEL_PATH = 'MediaPipe Models/hand_landmarker.task'
SCALER_PATH = 'Misc Models/scaler.pkl'
MODEL_STATE_PATH = 'Misc Models/hand_landmark_model_state.pth' # Using state dict path
NUM_CLASSES = 28
INPUT_SIZE = 63 # 21 landmarks * 3 coordinates (x, y, z)

# --- PyTorch Model Definition (Needed to load state_dict) ---
class ASLClassifier(nn.Module):
    """
    The model architecture must match the one used during training.
    """
    def __init__(self, input_size, num_classes):
        super(ASLClassifier, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Include dropout if used in training
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layer_stack(x)

# --- Utility Function: Landmark Drawing ---

def draw_landmarks_on_image(rgb_image, detection_result):
  """Draws MediaPipe hand landmarks and handedness label on the image."""
  annotated_image = np.copy(rgb_image)

  # Check for detected hands
  if not detection_result.hand_landmarks:
      return annotated_image

  # Loop through the detected hands to visualize.
  for hand_landmarks, handedness in zip(
        detection_result.hand_landmarks, detection_result.handedness
  ):
    # Draw the hand landmarks (using MediaPipe's drawing utilities)
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList(
        landmark=[
            landmark_pb2.NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in hand_landmarks
        ]
    )
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Draw handedness (left or right hand) label.
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    
    height, width, _ = annotated_image.shape
    text_x = int(min(x_coordinates) * width)
    # Position text slightly above the top landmark
    text_y = int(min(y_coordinates) * height) - MARGIN

    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

# --- Utility Function: Prediction Mapping ---

def map_prediction_to_char(predicted_value):
    """Maps the numeric prediction (0-27) back to the ASL character."""
    if predicted_value == 26:
        return 'del'
    elif predicted_value == 27:
        return 'space'
    elif 0 <= predicted_value < 26:
        return chr(predicted_value + ord('A'))
    return 'UNKNOWN'

# --- Setup: Load Models and Detector ---

print("1. Setting up models and detector...")
# Load PyTorch model (Best Practice: Load state_dict)
try:
    model = ASLClassifier(INPUT_SIZE, NUM_CLASSES)
    # The map_location handles loading a GPU-trained model onto a CPU, if necessary
    model.load_state_dict(torch.load(MODEL_STATE_PATH, map_location=torch.device('cpu')))
    model.eval() # Set model to evaluation mode
except FileNotFoundError:
    print(f"Error: PyTorch model state dict not found at {MODEL_STATE_PATH}. Please check path.")
    exit()

# Load the StandardScaler
try:
    with open(SCALER_PATH, 'rb') as f:
        scaler = pickle.load(f)
except FileNotFoundError:
    print(f"Error: Scaler not found at {SCALER_PATH}. Please run training script first.")
    exit()

# Setup MediaPipe Hand Landmarker
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# --- Real-Time Video Loop ---

print("2. Starting video stream...")
cv2.namedWindow("ASL Classification")
vc = cv2.VideoCapture(0)

if not vc.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Use a list to store previous predictions for display
previous_predictions = [] 
PRED_HISTORY_SIZE = 5 # Smoothing prediction by showing history

try:
    while vc.isOpened():
        rval, frame = vc.read()
        if not rval:
            break
        
        # 1. Image Preprocessing
        # Convert BGR (OpenCV default) to RGB (MediaPipe requirement)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # 2. MediaPipe Detection
        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(frame, detection_result)

        # 3. Prediction Pipeline
        if detection_result.hand_world_landmarks:
            # We assume the first detected hand is the one we want to classify
            landmarks = detection_result.hand_world_landmarks[0]
            
            # --- Efficient Landmark Extraction (NumPy-based) ---
            # Flatten the list of WorldLandmark objects (x, y, z) into a single 1D array
            single_lm_np = np.array([[lm.x, lm.y, lm.z] for lm in landmarks]).flatten()
            
            # Reshape, Scale, and Convert to Tensor
            single_lm_scaled = scaler.transform(single_lm_np.reshape(1, -1))
            single_lm_tensor = torch.tensor(single_lm_scaled).float()
            
            # Perform Inference (inside no_grad for efficiency)
            with torch.no_grad():
                output = model(single_lm_tensor)
                _, predicted = torch.max(output, 1)

            # Map to ASL Character
            pd_value = map_prediction_to_char(predicted.item())
            
            # Add to history
            previous_predictions.append(pd_value)
            if len(previous_predictions) > PRED_HISTORY_SIZE:
                previous_predictions.pop(0)

        # 4. Display Results
        
        # Display the most recent prediction
        if previous_predictions:
            display_text = previous_predictions[-1]
            cv2.putText(annotated_image, 
                        f"ASL: {display_text}", 
                        PREDICTION_TEXT_POS, 
                        cv2.FONT_HERSHEY_DUPLEX, 
                        FONT_SIZE + 0.5, # Slightly larger font
                        PREDICTION_TEXT_COLOR, 
                        FONT_THICKNESS + 1, 
                        cv2.LINE_AA)
        else:
             cv2.putText(annotated_image, 
                        "No Hand Detected", 
                        PREDICTION_TEXT_POS, 
                        cv2.FONT_HERSHEY_DUPLEX, 
                        FONT_SIZE + 0.5, 
                        (0, 0, 255), 
                        FONT_THICKNESS + 1, 
                        cv2.LINE_AA)

        # 5. Show Frame and Handle Exit
        cv2.imshow("ASL Classification", annotated_image)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    # Cleanup resources
    print("3. Releasing resources...")
    cv2.destroyAllWindows()
    vc.release()
