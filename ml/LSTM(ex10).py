import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
import matplotlib.pyplot as plt

# 1. Load and preprocess the data
max_features = 10000
maxlen = 100

(x_train, y_train), (x_test, y_test) = imdb.load_data(
    num_words=max_features
)

# Pad sequences to the same length
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)

# 2. Create the LSTM model
embedding_dim = 128

model = models.Sequential([
    layers.Input(shape=(maxlen,)),
    layers.Embedding(max_features, embedding_dim),
    layers.LSTM(128),
    layers.Dense(1, activation="sigmoid")
])

# 3. Compile the model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Display model architecture
model.summary()

# 4. Train the model
history = model.fit(
    x_train,
    y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)

# 5. Evaluate the model on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_acc * 100, "%")

# 6. Plot training and validation loss
plt.figure(figsize=(8, 6))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("LSTM Model Loss")
plt.legend()
plt.grid(True)
plt.show()

# 7. Plot training and validation accuracy
plt.figure(figsize=(8, 6))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("LSTM Model Accuracy")
plt.legend()
plt.grid(True)
plt.show()
