"""
step6_lstm.py
LSTM forecasting on stock data.
Run: python step6_lstm.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------- USER SETTINGS --------
CSV_FILE = "AAPL_history_2015-01-01_2024-12-31.csv"
TRAIN_RATIO = 0.8
LOOKBACK = 60
EPOCHS = 20
BATCH_SIZE = 32
# --------------------------------

# Load dataset
data = pd.read_csv(CSV_FILE, parse_dates=["Date"], index_col="Date")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]
close_data = data[['Close']].values

# Scale
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(close_data)

# Train-test split
train_size = int(len(scaled_data) * TRAIN_RATIO)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size - LOOKBACK:]

def create_sequences(data, lookback):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, LOOKBACK)
X_test, y_test = create_sequences(test_data, LOOKBACK)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

print("Training LSTM...")
model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# Predict
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1,1))

rmse = np.sqrt(mean_squared_error(y_test_rescaled, predictions))
print(f"LSTM Test RMSE: {rmse:.4f}")

# Plot
train = data[:train_size]
test = data[train_size:]
test['Predicted'] = predictions
plt.figure(figsize=(12,6))
plt.plot(train['Close'], label='Train')
plt.plot(test['Close'], label='Test', color='blue')
plt.plot(test['Predicted'], label='LSTM Forecast', color='red')
plt.legend()
plt.show()

# --- Save future forecast (30 days) ---
last_sequence = scaled_data[-LOOKBACK:]
current_input = last_sequence.reshape(1, LOOKBACK, 1)
future_predictions = []
for _ in range(30):
    next_pred = model.predict(current_input)[0,0]
    future_predictions.append(next_pred)
    current_input = np.append(current_input[:,1:,:], [[[next_pred]]], axis=1)

future_predictions_rescaled = scaler.inverse_transform(np.array(future_predictions).reshape(-1,1))
pd.DataFrame({
    "Date": pd.date_range(data.index[-1], periods=30, freq='B'),
    "Forecast": future_predictions_rescaled.flatten()
}).to_csv("forecast_lstm.csv", index=False)
print("Saved LSTM forecast to forecast_lstm.csv")
