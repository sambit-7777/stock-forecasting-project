"""
step7_model_comparison.py
Compare ARIMA, SARIMA, Prophet, and LSTM forecasts.
Run: python step7_model_comparison.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- RMSE values from previous steps ---
results = {
    'ARIMA': 69.2944,
    'SARIMA': 55.1133,
    'Prophet': 38.3315,
    'LSTM': 5.9821
}

# Convert to DataFrame
df_results = pd.DataFrame(list(results.items()), columns=['Model','RMSE'])

# Print results
print("Model Comparison (Lower RMSE is Better):")
print(df_results)

# --- Plot RMSE comparison ---
plt.figure(figsize=(8,5))
plt.bar(df_results['Model'], df_results['RMSE'], color=['orange','blue','green','red'])
plt.title("Model Comparison (Lower RMSE is Better)")
plt.ylabel("RMSE")
plt.show()
