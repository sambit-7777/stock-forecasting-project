"""
step5_prophet.py
Fit Prophet model on stock data.
Run: python step5_prophet.py
"""

import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error
import numpy as np

# -------- USER SETTINGS --------
CSV_FILE = "AAPL_history_2015-01-01_2024-12-31.csv"
TRAIN_RATIO = 0.8
# --------------------------------

# Load dataset
data = pd.read_csv(CSV_FILE, parse_dates=["Date"], index_col="Date")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

df = data.reset_index()[['Date', 'Close']]
df.columns = ['ds', 'y']

# Train-test split
train_size = int(len(df) * TRAIN_RATIO)
train, test = df.iloc[:train_size], df.iloc[train_size:]

# Fit Prophet
model = Prophet(daily_seasonality=True)
model.fit(train)

# Forecast test
future = model.make_future_dataframe(periods=len(test))
forecast = model.predict(future)
forecast_test = forecast.iloc[-len(test):][['ds','yhat']]

# RMSE
rmse = np.sqrt(mean_squared_error(test['y'], forecast_test['yhat']))
print(f"Prophet Test RMSE: {rmse:.4f}")

# Plots
model.plot(forecast); plt.show()
model.plot_components(forecast); plt.show()

plt.figure(figsize=(12,5))
plt.plot(train['ds'], train['y'], label='Train')
plt.plot(test['ds'], test['y'], label='Test', color='blue')
plt.plot(test['ds'], forecast_test['yhat'], label='Forecast', color='red')
plt.title("Prophet Forecast vs Actual")
plt.legend()
plt.show()

# --- Save future forecast (30 days) ---
future_30 = model.make_future_dataframe(periods=30)
forecast_30 = model.predict(future_30)

forecast_30[['ds','yhat']].tail(30).rename(columns={'ds':'Date','yhat':'Forecast'}) \
    .to_csv("forecast_prophet.csv", index=False)
print("Saved Prophet forecast to forecast_prophet.csv")
