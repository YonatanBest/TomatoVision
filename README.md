# TomatoVision

TomatoVision is a web application for automatic leaf image classification and disease detection using deep learning. It allows users to upload an image and receive predictions for:

1. **Leaf vs. Non-Leaf** (Autoencoder)
2. **Tomato vs. Non-Tomato Leaf** (Leaf Classifier)
3. **Tomato Leaf Disease Classification** (Disease Classifier)

## Features
- Upload any image and classify if it contains a leaf.
- If a leaf is detected, determine if it is a tomato leaf.
- If a tomato leaf is detected, identify the disease (or healthy) from 9 possible classes.
- User-friendly web interface built with Flask and Bootstrap.

## Model Files
Place the following pre-trained model files in the `model/` directory:
- `autoencoder.h5` (leaf vs. non-leaf)
- `leaf_classifier.h5` (tomato vs. non-tomato)
- `disease_classifer.h5` (tomato disease classifier)
- `tomato_classes.json` (list of disease class names)

## Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/YonatanBest/TomatoVision
   cd TomatoVision/app
   ```

2. **Install dependencies**
   ```bash
   pip install -r ../requirements.txt
   ```

3. **Add model files**
   Place the required `.h5` model files and `tomato_classes.json` in the `model/` directory as described above.

4. **Run the app**
   ```bash
   python app.py
   ```
   The app will be available at `http://127.0.0.1:5000/`.

## Usage
- Open the web app in your browser.
- Upload an image (leaf or non-leaf).
- View the classification results and disease prediction (if applicable).

## Disease Classes
The disease classifier predicts the following classes:
- Tomato_Bacterial_spot
- Tomato_Early_blight
- Tomato_Late_blight
- Tomato_Septoria_leaf_spot
- Tomato_Spider_mites_Two_spotted_spider_mite
- Tomato__Target_Spot
- Tomato__Tomato_YellowLeaf__Curl_Virus
- Tomato__Tomato_mosaic_virus
- Tomato_healthy

## Notes
- The threshold for leaf detection is set to 0.075 (can be adjusted in `app.py`).
- All models must be trained and exported as Keras `.h5` files.

## License
MIT License
