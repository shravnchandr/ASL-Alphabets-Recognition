import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader # Import for batch processing
from tqdm import tqdm # Import tqdm for cleaner training progress

# --- Configuration ---
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 1000
INPUT_SIZE = 63 # 21 landmarks * 3 coordinates (x, y, z)
NUM_CLASSES = 28 # A-Z (26) + 'del' (1) + 'space' (1)

# --- Data Loading and Preprocessing ---
print("1. Loading and Preprocessing Data...")
with open('mediapipe_hand_world_landmarks.pickle', 'rb') as handle:
    mediapipe_landmarks = pickle.load(handle)

X = []
y = []

# Map alphabet keys to numerical labels (0-27)
for alphabet, lm in mediapipe_landmarks.items():
    if len(alphabet) == 1:
        # A=0, B=1, ..., Z=25
        value = ord(alphabet) - ord('A')
    elif alphabet == 'del':
        value = 26
    elif alphabet == 'space':
        value = 27

    y.extend([value] * len(lm))
    X.extend(lm)

y = np.array(y)
X = np.array(X)

# Feature Scaling
scaler = StandardScaler()
X = scaler.fit_transform(X)

with open('Misc Models/scaler.pkl','wb') as f:
    pickle.dump(scaler, f)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42 # Added random_state for reproducibility
)

# Convert to PyTorch Tensors
X_train_tensor = torch.tensor(X_train).float()
y_train_tensor = torch.tensor(y_train).long()
X_test_tensor = torch.tensor(X_test).float()
y_test_tensor = torch.tensor(y_test).long()

# Create Dataset and DataLoader (Crucial for efficient training)
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- Model Definition ---
class ASLClassifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(ASLClassifier, self).__init__()
        self.layer_stack = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2), # Added Dropout for regularization
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layer_stack(x)

# Initialize Model and Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ASLClassifier(INPUT_SIZE, NUM_CLASSES).to(device)
print(f"Using device: {device}")
# 

[Image of a simple neural network diagram showing input, hidden layers, and output]


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training Loop (Optimized) ---

def calculate_accuracy(loader, model):
    """Calculates model accuracy over a DataLoader."""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

print("\n2. Starting Training...")
for epoch in range(NUM_EPOCHS):
    model.train() # Set model to training mode
    total_loss = 0
    
    # Iterate over batches using the DataLoader
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device) # Move data to device (e.g., GPU)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    avg_loss = total_loss / len(train_dataset)

    # Evaluation and Reporting
    if (epoch+1) % 100 == 0:
        train_acc = calculate_accuracy(train_loader, model)
        test_acc = calculate_accuracy(test_loader, model)

        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        
# --- Final Saving ---
print("\n3. Saving Model and Scaler...")
torch.save(model.state_dict(), 'Misc Models/hand_landmark_model_state.pth') # Save state_dict for best practice

print(f"✅ Training complete. Model saved to: hand_landmark_model_state.pth")