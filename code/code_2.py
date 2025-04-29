import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 1. Load and preprocess IMDB dataset
vocab_size = 10000
maxlen = 200

(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(num_words=vocab_size)
X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = keras.preprocessing.sequence.pad_sequences(X_test, maxlen=maxlen)

# 2. Build the model
model = keras.Sequential([
    keras.layers.Embedding(vocab_size, 32, input_length=maxlen),
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# 3. Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=512, validation_split=0.2)

# 5. Evaluate the model
loss, acc = model.evaluate(X_test, y_test)
print(f"\n✅ Test Accuracy: {acc * 100:.2f}%")

# 6. Plot accuracy
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()