from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import tensorflow as tf

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Define the classes
Classes = ["closed_eye", "open_eye"]

# Define image dimensions
img_size = 224

# Function to preprocess the image
def preprocess_image(image):
    img_array = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    backtorgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    resized_image = cv2.resize(backtorgb, (img_size, img_size))
    normalized_image = resized_image / 255.0
    return normalized_image

@app.route('/')
def home():
    return 'Welcome to the Drowsiness Detection API'

@app.route('/predict', methods=['POST'])
def predict():
    print(request.files)
    if 'image' not in request.files:
        return jsonify({'error': 'No image found'})

    image = request.files['image'].read()
    nparr = np.frombuffer(image, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processed_image = preprocess_image(img)
    processed_image = np.expand_dims(processed_image, axis=0)

    prediction = model.predict(processed_image)
    class_index = int(np.round(prediction[0][0]))

    result = {
        'class': Classes[class_index],
        'probability': float(prediction[0][0])
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0', port=5000)
