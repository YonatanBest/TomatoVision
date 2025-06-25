from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from werkzeug.utils import secure_filename
import os
import json

app = Flask(__name__)

# Load models
MODEL_DIR = os.path.join(os.path.dirname(__file__), '../model')
AUTOENCODER_PATH = os.path.join(MODEL_DIR, 'autoencoder.h5')
LEAF_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'leaf_classifier.h5')
DISEASE_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'disease_classifer.h5')
autoencoder = load_model(AUTOENCODER_PATH, compile=False)
leaf_classifier = load_model(LEAF_CLASSIFIER_PATH, compile=False)
disease_classifier = load_model(DISEASE_CLASSIFIER_PATH, compile=False)

# Load disease class names
with open(os.path.join(MODEL_DIR, 'tomato_classes.json'), 'r') as f:
    DISEASE_CLASS_NAMES = json.load(f)

LEAF_CLASS_NAMES = ['Non-Tomato', 'Tomato']

IMG_SIZE = (128, 128)
THRESHOLD = 0.075

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

            img = load_img(filepath, target_size=IMG_SIZE)
            img_arr = img_to_array(img)

            
            img_arr_auto = np.expand_dims(img_arr / 255.0, axis=0)
            img_arr_class = np.expand_dims(img_arr, axis=0)

            # Step 1: Autoencoder for leaf vs. non-leaf
            reconstructed = autoencoder.predict(img_arr_auto)
            reconstruction_error = np.mean(np.abs(img_arr_auto - reconstructed))
            
            if reconstruction_error > THRESHOLD:
                result = f"Not a leaf! (Reconstruction error: {reconstruction_error:.4f})"
            else:
                # Step 2: Tomato vs Non-Tomato classifier
                leaf_pred = leaf_classifier.predict(img_arr_class)
                pred_class_idx = np.argmax(leaf_pred)
                pred_class = LEAF_CLASS_NAMES[pred_class_idx]
                confidence = np.max(leaf_pred)
                if pred_class == 'Tomato':
                    # Step 3: Disease classifier (only for tomato leaves)
                    disease_pred = disease_classifier.predict(img_arr_class)
                    disease_idx = np.argmax(disease_pred)
                    disease_class = DISEASE_CLASS_NAMES[disease_idx]
                    disease_conf = np.max(disease_pred)
                    result = (
                        f"Leaf detected!<br>Classified as: <b>{pred_class} leaf</b>\n"
                        f"<br>Disease: <b>{disease_class.replace('_', ' ')}</b>\n"
                        f"<br>Disease confidence: {disease_conf:.2%}"
                        f"<br>Leaf confidence: {confidence:.2%}"
                        f"<br>Reconstruction error: {reconstruction_error:.4f}"
                    )
                else:
                    result = (
                        f"Leaf detected!<br>Classified as: <b>{pred_class} leaf</b>\n"
                        f"<br>Confidence: {confidence:.2%}"
                        f"<br>Reconstruction error: {reconstruction_error:.4f}"
                    )
    
    return render_template('index.html', result=result, error=error, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)