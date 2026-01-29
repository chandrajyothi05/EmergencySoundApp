import librosa
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import json

# Path to dataset
DATASET_PATH = "dataset"

# Define which classes are emergency
emergency_classes = ["crying_baby", "glass_breaking", "siren"]

# Function to extract MFCC features
def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)

X = []
y = []

# Automatically detect subfolders (each = a class)
class_folders = [f for f in os.listdir(DATASET_PATH) if os.path.isdir(os.path.join(DATASET_PATH, f))]
labels = {folder_name: idx for idx, folder_name in enumerate(class_folders)}
print("Label map:", labels)

# Extract features from audio in each class folder
for class_name, class_idx in labels.items():
    folder_path = os.path.join(DATASET_PATH, class_name)
    files = [f for f in os.listdir(folder_path) if f.endswith(".wav")]
    print(f"{class_name}: {len(files)} files found")
    for file in files:
        file_path = os.path.join(folder_path, file)
        features = extract_features(file_path)
        X.append(features)
        y.append(class_idx)

X = np.array(X)
y = np.array(y)

print("Total samples:", len(X))
if len(X) == 0:
    print("No audio files found! Check dataset folders and .wav extensions.")
    exit()

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Test accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save trained model
joblib.dump(model, "emergency_sound_model.pkl")
print("Model saved as emergency_sound_model.pkl")

# Save label map + emergency class list
with open("label_map.json", "w") as f:
    json.dump({
        "labels": labels,
        "emergency_classes": emergency_classes
    }, f)
print("Label map + emergency class list saved as label_map.json")