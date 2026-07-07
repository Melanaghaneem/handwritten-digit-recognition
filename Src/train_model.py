from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


# Load dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("Dataset loaded successfully.")


# Normalize images
X_train = X_train / 255.0
X_test = X_test / 255.0

print("Images normalized.")


# One-hot encoding
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

print("Labels encoded successfully.")


# Build neural network model
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(10, activation="softmax")
])


# Display model structure
model.summary()

# Compile the model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model compiled successfully.")


# Train the model
history = model.fit(
    X_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

print("Model training completed.")


# Evaluate the model
test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test
)

print("Test Accuracy:", test_accuracy)