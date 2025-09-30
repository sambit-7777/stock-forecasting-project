"""
step1_fetch_data.py
Fetch historical stock data using yfinance, show basic info, drop missing rows and save CSV.
Run: python step1_fetch_data.py
"""

import os
from datetime import datetime
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# --------- USER SETTINGS (edit these) ---------
TICKER = "AAPL"            # Example: 'AAPL' (Apple). For NSE tickers use 'TCS.NS' or 'RELIANCE.NS'
START = "2015-01-01"
END = "2024-12-31"         # Set to a recent date; you can update to today's date
OUTPUT_CSV = f"{TICKER}_history_{START}_{END}.csv"
# ----------------------------------------------


def fetch_save(ticker, start, end, out_csv):
    """Download, do quick checks, drop NA rows and save CSV. Returns cleaned DataFrame."""
    print(f"Downloading {ticker} from {start} to {end} ...")
    df = yf.download(ticker, start=start, end=end, progress=False)

    # 🔧 Flatten MultiIndex columns if needed
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Basic checks
    if df is None or df.empty:
        raise RuntimeError("No data was downloaded. Check the ticker symbol and your internet connection.")

    print("Downloaded rows:", len(df))
    print("Columns:", df.columns.tolist())
    print("First 5 rows:\n", df.head())

    # Count missing values per column
    print("\nMissing values per column:\n", df.isna().sum())

    # Drop any rows with missing values (simple hygiene step)
    df_clean = df.dropna()
    df_clean.to_csv(out_csv)
    print(f"Saved cleaned CSV to {out_csv}")
    return df_clean


if __name__ == "__main__":
    data = fetch_save(TICKER, START, END, OUTPUT_CSV)

    # Quick plot of closing price to verify visually
    plt.figure(figsize=(10, 4))
    plt.plot(data.index, data['Close'], label='Close', color='blue')
    plt.title(f"{TICKER} Close Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
