import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math

# 1. Load data
data = yf.download('GOOG', start='2015-01-01', end='2023-12-31')[['Close']]

# 2. Normalize
scaler = MinMaxScaler()
scaled = scaler.fit_transform(data)

# 3. Create sequences
def create_sequences(data, seq_len=60):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i - seq_len:i])
        y.append(data[i])
    return np.array(X), np.array(y)

X, y = create_sequences(scaled)

# 4. Reshape and split
X = X.reshape(X.shape[0], X.shape[1], 1)
split = int(0.8 * len(X))
X_train, y_train = X[:split], y[:split]
X_test, y_test = X[split:], y[split:]

# 5. Build model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X.shape[1], 1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 6. Train
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# 7. Predict
preds = model.predict(X_test)
preds_rescaled = scaler.inverse_transform(preds)
actual = scaler.inverse_transform(y_test)

# 8. Plot
plt.figure(figsize=(12, 6))
plt.plot(actual, label='Actual')
plt.plot(preds_rescaled, label='Predicted')
plt.title('GOOG Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()

# 9. RMSE
rmse = math.sqrt(mean_squared_error(actual, preds_rescaled))
print(f"✅ RMSE: {rmse:.2f}")