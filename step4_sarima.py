"""
step4_sarima.py
Fit SARIMA model on stock data.
Run: python step4_sarima.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error

# -------- USER SETTINGS --------
CSV_FILE = "AAPL_history_2015-01-01_2024-12-31.csv"
TRAIN_RATIO = 0.8
ORDER = (1,1,1)
SEASONAL_ORDER = (1,1,1,12)
# --------------------------------

# Load dataset
data = pd.read_csv(CSV_FILE, parse_dates=["Date"], index_col="Date")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]
ts = data['Close']

# Train-test split
train_size = int(len(ts) * TRAIN_RATIO)
train, test = ts[:train_size], ts[train_size:]
print(f"Train size: {len(train)}, Test size: {len(test)}")

# Fit SARIMA
print(f"Fitting SARIMA{ORDER}x{SEASONAL_ORDER}...")
model = SARIMAX(train, order=ORDER, seasonal_order=SEASONAL_ORDER,
                enforce_stationarity=False, enforce_invertibility=False)
model_fit = model.fit(disp=False)
print(model_fit.summary())

# Forecast test set
forecast = model_fit.forecast(steps=len(test))
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"Test RMSE: {rmse:.4f}")

# Plot
plt.figure(figsize=(12,5))
plt.plot(train.index, train, label='Train')
plt.plot(test.index, test, label='Test', color='blue')
plt.plot(test.index, forecast, label='Forecast', color='red')
plt.title(f"SARIMA{ORDER}x{SEASONAL_ORDER} Forecast vs Actual")
plt.legend()
plt.show()

# --- Save future forecast (30 days) ---
future_model = SARIMAX(ts, order=ORDER, seasonal_order=SEASONAL_ORDER,
                       enforce_stationarity=False, enforce_invertibility=False)
future_fit = future_model.fit(disp=False)
future_forecast = future_fit.forecast(steps=30)

pd.DataFrame({
    "Date": pd.date_range(ts.index[-1], periods=30, freq='B'),
    "Forecast": future_forecast
}).to_csv("forecast_sarima.csv", index=False)
print("Saved SARIMA forecast to forecast_sarima.csv")
