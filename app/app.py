from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import img_to_array, load_img
from werkzeug.utils import secure_filename
import os
import json
import google.generativeai as genai
from PIL import Image

app = Flask(__name__)

MODEL_DIR = os.path.join(os.path.dirname(__file__), '../model')
AUTOENCODER_GRAY_PATH = os.path.join(MODEL_DIR, 'autoencoder.h5')
AUTOENCODER_COLOR_PATH = os.path.join(MODEL_DIR, 'autoencoder_color.h5')
LEAF_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'leaf_classifier.h5')
DISEASE_CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'disease_classifer.h5')
autoencoder_gray = load_model(AUTOENCODER_GRAY_PATH, compile=False)
autoencoder_color = load_model(AUTOENCODER_COLOR_PATH, compile=False)
leaf_classifier = load_model(LEAF_CLASSIFIER_PATH, compile=False)
disease_classifier = load_model(DISEASE_CLASSIFIER_PATH, compile=False)

with open(os.path.join(MODEL_DIR, 'tomato_classes.json'), 'r') as f:
    DISEASE_CLASS_NAMES = json.load(f)

LEAF_CLASS_NAMES = ['Non-Tomato', 'Tomato']

IMG_SIZE = (128, 128)
THRESHOLD = 0.067
COLOR_THRESHOLD = 0.067

GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')

def analyze_with_gemini(image_path):
    if not GEMINI_API_KEY:
        return False, "Error: Gemini API key not configured. Please set the GEMINI_API_KEY environment variable."
    
    try:
        image = Image.open(image_path)
        prompt = """Analyze this image and determine if it shows a leaf or not. 
        Is this a leaf? (Yes/No)"""
        
        response = gemini_model.generate_content([prompt, image])
        is_leaf = "yes" in response.text.lower()
        return is_leaf, response.text
    except Exception as e:
        return False, f"Error analyzing image with Gemini: {str(e)}"

def process_classification(img_arr_class):
    """Process tomato and disease classification using local models."""
    leaf_pred = leaf_classifier.predict(img_arr_class)
    pred_class_idx = np.argmax(leaf_pred)
    pred_class = LEAF_CLASS_NAMES[pred_class_idx]
    confidence = np.max(leaf_pred)
    
    if pred_class == 'Tomato':
        disease_pred = disease_classifier.predict(img_arr_class)
        disease_idx = np.argmax(disease_pred)
        disease_class = DISEASE_CLASS_NAMES[disease_idx]
        disease_conf = np.max(disease_pred)
        return True, pred_class, confidence, disease_class, disease_conf
    else:
        return False, pred_class, confidence, None, None

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
        
        model_choice = request.form.get('model_choice', 'autoencoder_gray')
        
        if file:
            static_dir = os.path.join(os.path.dirname(__file__), 'static')
            os.makedirs(static_dir, exist_ok=True)
            filename = secure_filename(file.filename)
            filepath = os.path.join(static_dir, filename)
            file.save(filepath)

            img_color = load_img(filepath, target_size=IMG_SIZE, color_mode='rgb')
            img_arr_color = img_to_array(img_color)
            img_arr_class = np.expand_dims(img_arr_color, axis=0)

            is_leaf = False
            if model_choice == 'gemini':
                is_leaf, gemini_result = analyze_with_gemini(filepath)
                if gemini_result.startswith('Error'):
                    error = gemini_result
                    return render_template('index.html', result=result, error=error, filename=filename)
                elif not is_leaf:
                    result = gemini_result
                else:
                    is_tomato, pred_class, confidence, disease_class, disease_conf = process_classification(img_arr_class)
                    if is_tomato:
                        result = (
                            f"{gemini_result}<br><br>"
                            f"Further Analysis:<br>"
                            f"Classified as: <b>{pred_class} leaf</b>\n"
                            f"<br>Disease: <b>{disease_class.replace('_', ' ')}</b>\n"
                            f"<br>Disease confidence: {disease_conf:.2%}"
                            f"<br>Leaf confidence: {confidence:.2%}"
                        )
                    else:
                        result = (
                            f"{gemini_result}<br><br>"
                            f"Further Analysis:<br>"
                            f"Classified as: <b>{pred_class} leaf</b>\n"
                            f"<br>Confidence: {confidence:.2%}"
                        )
            
            elif model_choice == 'autoencoder_color':
                img_arr_auto = np.expand_dims(img_arr_color / 255.0, axis=0)
                reconstructed = autoencoder_color.predict(img_arr_auto)
                reconstruction_error = np.mean(np.abs(img_arr_auto - reconstructed))
                
                if reconstruction_error > COLOR_THRESHOLD:
                    result = f"Not a leaf! (Color reconstruction error: {reconstruction_error:.4f})"
                else:
                    is_tomato, pred_class, confidence, disease_class, disease_conf = process_classification(img_arr_class)
                    if is_tomato:
                        result = (
                            f"Leaf detected!<br>Classified as: <b>{pred_class} leaf</b>\n"
                            f"<br>Disease: <b>{disease_class.replace('_', ' ')}</b>\n"
                            f"<br>Disease confidence: {disease_conf:.2%}"
                            f"<br>Leaf confidence: {confidence:.2%}"
                            f"<br>Color reconstruction error: {reconstruction_error:.4f}"
                        )
                    else:
                        result = (
                            f"Leaf detected!<br>Classified as: <b>{pred_class} leaf</b>\n"
                            f"<br>Confidence: {confidence:.2%}"
                            f"<br>Color reconstruction error: {reconstruction_error:.4f}"
                        )
            
            else:
                img_gray = load_img(filepath, target_size=IMG_SIZE, color_mode='grayscale')
                img_arr_gray = img_to_array(img_gray)
                img_arr_auto = np.expand_dims(img_arr_gray / 255.0, axis=0)

                reconstructed = autoencoder_gray.predict(img_arr_auto)
                reconstruction_error = np.mean(np.abs(img_arr_auto - reconstructed))
                
                if reconstruction_error > THRESHOLD:
                    result = f"Not a leaf! (Grayscale reconstruction error: {reconstruction_error:.4f})"
                else:
                    is_tomato, pred_class, confidence, disease_class, disease_conf = process_classification(img_arr_class)
                    if is_tomato:
                        result = (
                            f"Leaf detected!<br>Classified as: <b>{pred_class} leaf</b>\n"
                            f"<br>Disease: <b>{disease_class.replace('_', ' ')}</b>\n"
                            f"<br>Disease confidence: {disease_conf:.2%}"
                            f"<br>Leaf confidence: {confidence:.2%}"
                            f"<br>Grayscale reconstruction error: {reconstruction_error:.4f}"
                        )
                    else:
                        result = (
                            f"Leaf detected!<br>Classified as: <b>{pred_class} leaf</b>\n"
                            f"<br>Confidence: {confidence:.2%}"
                            f"<br>Grayscale reconstruction error: {reconstruction_error:.4f}"
                        )
    
    return render_template('index.html', result=result, error=error, filename=filename)

if __name__ == '__main__':
    app.run(debug=True)