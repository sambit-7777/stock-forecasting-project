"""
step8_dashboard.py
Visualization Dashboard for Stock Forecasting.
Run: python step8_dashboard.py
"""

import pandas as pd
import plotly.graph_objects as go

# --- Load historical data ---
CSV_FILE = "AAPL_history_2015-01-01_2024-12-31.csv"
data = pd.read_csv(CSV_FILE, parse_dates=["Date"], index_col="Date")

if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] for col in data.columns]

# Dummy forecast data (replace with real forecast outputs saved from earlier steps)
# For now, let's simulate by shifting the close price
future_dates = pd.date_range(data.index[-1], periods=30, freq='B')

forecast_arima = data['Close'].iloc[-30:].values * 1.01
forecast_sarima = data['Close'].iloc[-30:].values * 1.02
forecast_prophet = data['Close'].iloc[-30:].values * 1.03
forecast_lstm = data['Close'].iloc[-30:].values * 1.04

# --- Plotly figure ---
fig = go.Figure()

# Historical
fig.add_trace(go.Scatter(x=data.index, y=data['Close'],
                         mode='lines', name='Historical'))

# Forecasts
fig.add_trace(go.Scatter(x=future_dates, y=forecast_arima,
                         mode='lines+markers', name='ARIMA Forecast'))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_sarima,
                         mode='lines+markers', name='SARIMA Forecast'))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_prophet,
                         mode='lines+markers', name='Prophet Forecast'))
fig.add_trace(go.Scatter(x=future_dates, y=forecast_lstm,
                         mode='lines+markers', name='LSTM Forecast'))

# Layout
fig.update_layout(
    title="Stock Price Forecast Comparison",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark"
)

fig.show()
