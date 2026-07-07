from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Dataset loaded successfully.")

# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Images normalized.")

# Convert labels to One-Hot Encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Labels encoded successfully.")

print("Training images:", X_train.shape)
print("Training labels:", y_train.shape)

print("Testing images:", X_test.shape)
print("Testing labels:", y_test.shape)