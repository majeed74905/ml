import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt

# Load MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Preprocess the images
x_train = x_train.reshape((60000, 784)).astype("float32") / 255.0
x_test = x_test.reshape((10000, 784)).astype("float32") / 255.0

# Define encoding dimension
encoding_dim = 32

# Input layer
input_img = tf.keras.Input(shape=(784,))

# Encoder
encoded = layers.Dense(encoding_dim, activation="relu")(input_img)

# Decoder
decoded = layers.Dense(784, activation="sigmoid")(encoded)

# Build autoencoder model
autoencoder = models.Model(input_img, decoded)

# Compile the model
autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy"
)

# Display model summary
autoencoder.summary()

# Train the autoencoder
history = autoencoder.fit(
    x_train,
    x_train,
    epochs=5,
    batch_size=256,
    shuffle=True,
    validation_data=(x_test, x_test)
)

# Reconstruct test images
decoded_imgs = autoencoder.predict(x_test)

# Display original and reconstructed images
n = 10

plt.figure(figsize=(20, 4))

for i in range(n):

    # Original images
    plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    # Reconstructed images
    plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.tight_layout()
plt.show()

# Plot training and validation loss
plt.figure(figsize=(8, 5))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Autoencoder Loss")
plt.legend()
plt.grid(True)
plt.show()
