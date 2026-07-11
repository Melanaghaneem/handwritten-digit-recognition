import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
load_model = tf.keras.models.load_model
from PIL import Image

# 1. Load the trained Keras model into memory when the server starts
MODEL_PATH = os.path.join(os.path.dirname(
    __file__), 'digit_model.keras')  
model = load_model(MODEL_PATH)


def predict_digit(image_path):
    """
    Processes the uploaded image and passes it to the Keras model.
    """
    # 2. Preprocess the image to match MNIST standards (28x28 grayscale)
    # Open image and convert to Grayscale ('L')
    img = Image.open(image_path).convert('L')

    # Resize to 28x28 pixels as required by most handwritten digit models
    img = img.resize((28, 28))

    # Convert image to a numpy array
    img_array = np.array(img)

    # 3. Handle Inversion (Optional but important)
    # MNIST models expect a white digit on a black background.
    # If your frontend sends a black digit on a white background, uncomment the line below:
    # img_array = 255 - img_array

    # 4. Normalize pixel values to be between 0.0 and 1.0
    img_array = img_array / 255.0

    # 5. Reshape array to match the model's expected input shape
    # Most models expect: (batch_size, width, height, channels) -> (1, 28, 28, 1)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = np.expand_dims(img_array, axis=-1)

    # 6. Run the prediction
    predictions = model.predict(img_array)

    # Extract the digit with the highest probability
    predicted_digit = int(np.argmax(predictions[0]))

    # Extract the confidence score of that specific prediction
    confidence = float(np.max(predictions[0]))

    return predicted_digit, confidence
