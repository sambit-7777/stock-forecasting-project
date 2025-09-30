"""
step2_eda.py
Perform data cleaning and exploratory data analysis (EDA) on the stock dataset.
Run: python step2_eda.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

# -------- USER SETTINGS --------
CSV_FILE = "AAPL_history_2015-01-01_2024-12-31.csv"  # use the file you saved in Step 1
# --------------------------------

# Load dataset
data = pd.read_csv(CSV_FILE, parse_dates=["Date"], index_col="Date")

# 🔧 Flatten multi-index columns if necessary
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

print("Data shape:", data.shape)
print("Columns:", data.columns.tolist())
print(data.head())

# --- Plot Close price ---
plt.figure(figsize=(10, 4))
plt.plot(data.index, data['Close'], color='blue')
plt.title("Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Volume traded ---
plt.figure(figsize=(10, 4))
plt.plot(data.index, data['Volume'], color='orange')
plt.title("Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Feature: Daily Returns ---
data['Returns'] = data['Close'].pct_change()

plt.figure(figsize=(10, 4))
plt.plot(data.index, data['Returns'], color='green')
plt.title("Daily Returns")
plt.xlabel("Date")
plt.ylabel("Return")
plt.grid(True)
plt.tight_layout()
plt.show()

print("Returns summary:")
print(data['Returns'].describe())

# --- Feature: Rolling Mean & Std ---
data['RollingMean30'] = data['Close'].rolling(window=30).mean()
data['RollingStd30'] = data['Close'].rolling(window=30).std()

plt.figure(figsize=(12, 5))
plt.plot(data['Close'], label='Close')
plt.plot(data['RollingMean30'], label='30-Day Mean', color='red')
plt.fill_between(data.index,
                 data['RollingMean30'] - 2 * data['RollingStd30'],
                 data['RollingMean30'] + 2 * data['RollingStd30'],
                 color='lightgray', alpha=0.5)
plt.legend()
plt.title("Close with 30-Day Rolling Mean & Std")
plt.show()

# --- Trend and Seasonality Decomposition ---
result = seasonal_decompose(data['Close'], model='multiplicative', period=252)  # ~252 trading days per year
result.plot()
plt.show()

# --- Stationarity Test (ADF test) ---
print("Checking stationarity of Close series...")
adf_result = adfuller(data['Close'].dropna())
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
for key, value in adf_result[4].items():
    print('Critical Value (%s): %.3f' % (key, value))

if adf_result[1] <= 0.05:
    print("✅ Series is stationary (good for ARIMA)")
else:
    print("⚠️ Series is NOT stationary, differencing may be needed")
