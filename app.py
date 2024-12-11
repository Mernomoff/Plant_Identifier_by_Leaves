from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import numpy as np
import pickle
import os
import io
from PIL import Image

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join("model", "plant_leaf_classifier.h5")
CLASS_NAMES_PATH = os.path.join("model", "class_names.pkl")
IMG_HEIGHT, IMG_WIDTH = 224, 224

model = load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, 'rb') as f:
    class_names = pickle.load(f)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Изображение не было загружено"}), 400

    file = request.files["file"]

    try:
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image = image.resize((IMG_HEIGHT, IMG_WIDTH))
        image_array = img_to_array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        predictions = model.predict(image_array)
        predicted_class_idx = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_names[predicted_class_idx]

        return jsonify({
            "predicted_class": predicted_class_name,
            "confidence": float(np.max(predictions))
        })

    except Exception as e:
        print(f"Ошибка предсказания: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
