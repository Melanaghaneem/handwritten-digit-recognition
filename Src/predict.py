import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image


# Load trained model
model = load_model("model/digit_model.keras")


def preprocess_image(image_path):
    """
    Prepare input image for the model
    """

    # Open image and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Resize image to MNIST size
    image = image.resize((28, 28))

    # Convert image to numpy array
    image = np.array(image)

    # Normalize pixels
    image = image / 255.0

    # Add batch dimension
    image = image.reshape(1, 28, 28)

    return image


def predict_digit(image_path):

    processed_image = preprocess_image(image_path)

    prediction = model.predict(processed_image)

    digit = np.argmax(prediction)

    confidence = np.max(prediction)

    return digit, confidence


# Test prediction
if __name__ == "__main__":

    image_path = "test_digit.jpg"

    digit, confidence = predict_digit(image_path)

    print("Predicted Digit:", digit)
    print("Confidence:", confidence)