from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../model')
AUTOENCODER_PATH = os.path.join(MODEL_DIR, 'autoencoder.h5')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'leaf_classifier.h5')
autoencoder = load_model(AUTOENCODER_PATH, compile=False)
classifier = load_model(CLASSIFIER_PATH, compile=False)

IMG_SIZE = (128, 128)
THRESHOLD = 0.07  # Set this based on your validation experiments

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    error = None
    filename = None
    if request.method == 'POST':
        if 'image' not in request.files:
            error = 'No file part'
            return render_template('index.html', result=result, error=error)
        file = request.files['image']
        if file.filename == '':
            error = 'No selected file'
            return render_template('index.html', result=result, error=error)
        if file:
            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            os.makedirs(static_dir, exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(static_dir, filename)
            file.save(filepath)
            # Preprocess the image
            img = load_img(filepath, target_size=IMG_SIZE)
            img_arr = img_to_array(img) / 255.0
            img_arr = np.expand_dims(img_arr, axis=0)  # (1, 128, 128, 3)
            # Step 1: Autoencoder for leaf vs. non-leaf
            reconstructed = autoencoder.predict(img_arr)
            reconstruction_error = np.mean(np.abs(img_arr - reconstructed))
            if reconstruction_error > THRESHOLD:
                result = f"Not a leaf! (Reconstruction error: {reconstruction_error:.4f})"
            else:
                # Step 2: Classifier for tomato vs. non-tomato
                pred = classifier.predict(img_arr)
                is_tomato = pred[0][0] > 0.5
                label = 'Tomato leaf' if is_tomato else 'Non-tomato leaf'
                confidence = pred[0][0] if is_tomato else 1 - pred[0][0]
                result = f"Leaf detected! Classified as: <b>{label}</b> (Confidence: {confidence:.2%}, Reconstruction error: {reconstruction_error:.4f})"
    return render_template('index.html', result=result, error=error, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)
