from flask import Flask, request, jsonify
import librosa
import numpy as np
import joblib
import json

app = Flask(__name__)

model = joblib.load("emergency_sound_model.pkl")

with open("label_map.json") as f:
    data = json.load(f)

label_map = data["labels"]
emergency_classes = data["emergency_classes"]

# number → class name
id_to_class = {v: k for k, v in label_map.items()}


def extract_features(file_path):
    audio, sr = librosa.load(file_path, duration=3)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0)


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files["file"]
    file.save("temp.wav")

    features = extract_features("temp.wav").reshape(1, -1)
    prediction = int(model.predict(features)[0])

    class_name = id_to_class[prediction]
    is_emergency = class_name in emergency_classes

    return jsonify({
        "class": class_name,
        "emergency": is_emergency
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)