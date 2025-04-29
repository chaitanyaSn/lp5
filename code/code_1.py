import tensorflow as tf
from tensorflow.keras import layers, datasets
import matplotlib.pyplot as plt

# 1. Load and normalize data
(train_X, train_y), (test_X, test_y) = datasets.boston_housing.load_data()
mean, std = train_X.mean(axis=0), train_X.std(axis=0)
train_X = (train_X - mean) / std
test_X = (test_X - mean) / std

# 2. Build & compile model
model = tf.keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(train_X.shape[1],)),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# 3. Train
history = model.fit(train_X, train_y, epochs=100, validation_split=0.2, verbose=0)

# 4. Evaluate
loss, mae = model.evaluate(test_X, test_y, verbose=0)
print(f"\n✅ Test MAE: {mae:.2f}")

# 5. Plot loss
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('MSE Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()