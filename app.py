"""
app.py
Streamlit dashboard for stock forecasting.
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
CSV_FILE = "AAPL_history_2015-01-01_2024-12-31.csv"
data = pd.read_csv(CSV_FILE, parse_dates=["Date"], index_col="Date")
if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# Title
st.title("📈 Stock Forecasting Dashboard")
st.write("Compare ARIMA, SARIMA, Prophet, and LSTM forecasts.")

# Sidebar
model_choice = st.sidebar.selectbox("Select Model", ("ARIMA", "SARIMA", "Prophet", "LSTM"))

# Historical data
st.subheader("Historical Data")
st.line_chart(data['Close'])

# Load real forecasts
if model_choice == "ARIMA":
    forecast_df = pd.read_csv("forecast_arima.csv")
elif model_choice == "SARIMA":
    forecast_df = pd.read_csv("forecast_sarima.csv")
elif model_choice == "Prophet":
    forecast_df = pd.read_csv("forecast_prophet.csv")
else:  # LSTM
    forecast_df = pd.read_csv("forecast_lstm.csv")

future_dates = pd.to_datetime(forecast_df['Date'])
forecast = forecast_df['Forecast'].values

# Plot
st.subheader(f"{model_choice} Forecast (Next 30 Days)")
fig, ax = plt.subplots(figsize=(10,5))
ax.plot(data.index, data['Close'], label='Historical')
ax.plot(future_dates, forecast, label=f'{model_choice} Forecast', color='red')
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
st.pyplot(fig)

# RMSE table
st.subheader("Model Performance (RMSE)")
rmse_results = {
    'ARIMA': 69.2944,
    'SARIMA': 55.1133,
    'Prophet': 38.3315,
    'LSTM': 5.9821
}
st.table(pd.DataFrame(list(rmse_results.items()), columns=['Model','RMSE']))
