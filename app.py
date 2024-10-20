import tensorflow as tf
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta
import os
import traceback
import logging
import sys
from functools import lru_cache

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

# Global variables
MODEL_PATH = "Indian_Fruit_98.25.h5"
img_size = (224, 224)
model = None

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logger.info("GPU memory growth enabled")
    except RuntimeError as e:
        logger.error(f"Error configuring GPU: {e}")

# Load model
try:
    logger.info("Loading model...")
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
    model = None

class_names = [
    "Apple_Bad", "Apple_Good", "Apple_mixed",
    "Banana_Bad", "Banana_Good", "Banana_mixed",
    "Guava_Bad", "Guava_Good", "Guava_mixed",
    "Lemon_mixed",
    "Lime_Bad", "Lime_Good",
    "Orange_Bad", "Orange_Good", "Orange_mixed",
    "Pomegranate_Bad", "Pomegranate_Good", "Pomegranate_mixed"
]

shelf_life = {
    "Apple": {"shelf": "1-2 days", "refrigerator": "3 weeks", "freezer": "8 months (cooked)"},
    "Banana": {"shelf": "Until ripe", "refrigerator": "2 days (skin will blacken)",
               "freezer": "1 month (whole peeled)"},
    "Guava": {"shelf": "3-5 days", "refrigerator": "1 week", "freezer": "Do not freeze"},
    "Lemon": {"shelf": "10 days", "refrigerator": "1-2 weeks", "freezer": "Do not freeze"},
    "Lime": {"shelf": "10 days", "refrigerator": "1-2 weeks", "freezer": "Do not freeze"},
    "Orange": {"shelf": "10 days", "refrigerator": "1-2 weeks", "freezer": "Do not freeze"},
    "Pomegranate": {"shelf": "1-2 days", "refrigerator": "3-4 days", "freezer": "Balls, 1 month"}
}


# Preprocessing function with caching
@lru_cache(maxsize=32)
def preprocess_for_model(image_bytes):
    try:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        image = cv2.resize(image, img_size)
        image = preprocess_input(image)
        return np.expand_dims(image, axis=0)
    except Exception as e:
        logger.error(f"Error in preprocess_for_model: {e}")
        raise


# Prediction function
@tf.function
def predict(input_tensor):
    return model(input_tensor, training=False)


def get_shelf_life_info(fruit_class):
    try:
        fruit_type = fruit_class.split('_')[0]
        condition = fruit_class.split('_')[1]

        if fruit_type in shelf_life:
            info = shelf_life[fruit_type]
            if condition == "Bad":
                shelf_days = min(1, int(info["refrigerator"].split()[0]))
            elif condition == "mixed":
                shelf_days = max(1, int(int(info["refrigerator"].split()[0]) * 0.7))
            else:  # "Good"
                shelf_days = int(info["refrigerator"].split()[0])

            return {
                "shelf": info["shelf"],
                "refrigerator": f"{shelf_days} days",
                "freezer": info["freezer"],
                "estimated_days": shelf_days
            }
    except Exception as e:
        logger.error(f"Error in get_shelf_life_info: {e}")

    return {
        "shelf": "Unknown",
        "refrigerator": "Unknown",
        "freezer": "Unknown",
        "estimated_days": 3
    }


def process_image(image_bytes):
    try:
        preprocessed_img = preprocess_for_model(image_bytes)
        predictions = predict(preprocessed_img)
        predicted_class = class_names[np.argmax(predictions[0])]
        confidence = float(np.max(predictions[0]))

        shelf_life_info = get_shelf_life_info(predicted_class)
        expiry_date = datetime.now() + timedelta(days=shelf_life_info["estimated_days"])

        return {
            "fruit_class": predicted_class,
            "confidence": confidence,
            "shelf_life": shelf_life_info,
            "expiry_date": expiry_date.strftime('%Y-%m-%d')
        }
    except Exception as e:
        logger.error(f"Error in process_image: {e}")
        raise


@app.route('/test', methods=['POST'])
def test_endpoint():
    try:
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        return jsonify({
            "message": "Image received successfully",
            "filename": file.filename,
            "model_loaded": model is not None
        })
    except Exception as e:
        logger.error(f"Error in test_endpoint: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/detect_fruit', methods=['POST'])
def detect_fruit():
    try:
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        if 'image' not in request.files:
            return jsonify({"error": "No image file provided"}), 400

        file = request.files['image']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        image_bytes = file.read()
        result = process_image(image_bytes)
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in detect_fruit: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Welcome to the Fruit Detection API"}), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
