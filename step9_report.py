"""
step9_report.py
Generate final report and save results.
Run: python step9_report.py
"""

import pandas as pd
import matplotlib.pyplot as plt

# --- RMSE results (from Step 7) ---
results = {
    'ARIMA': 69.2944,
    'SARIMA': 55.1133,
    'Prophet': 38.3315,
    'LSTM': 5.9821
}

# Convert to DataFrame
df_results = pd.DataFrame(list(results.items()), columns=['Model','RMSE'])

# Save RMSE comparison to CSV
df_results.to_csv("model_comparison_rmse.csv", index=False)
print("Saved RMSE comparison to model_comparison_rmse.csv")

# Plot RMSE comparison
plt.figure(figsize=(8,5))
plt.bar(df_results['Model'], df_results['RMSE'], color=['orange','blue','green','red'])
plt.title("Model Comparison (Lower RMSE is Better)")
plt.ylabel("RMSE")
plt.savefig("model_comparison_rmse.png")
plt.show()

# --- Generate summary report ---
with open("final_report.txt", "w") as f:
    f.write("Time Series Stock Forecasting Report\n")
    f.write("=====================================\n\n")
    f.write("Models evaluated: ARIMA, SARIMA, Prophet, LSTM\n\n")
    f.write("RMSE Results:\n")
    for model, rmse in results.items():
        f.write(f" - {model}: {rmse:.4f}\n")
    
    f.write("\nKey Findings:\n")
    f.write("1. ARIMA performed the worst with RMSE ~69.\n")
    f.write("2. SARIMA improved results slightly (RMSE ~55).\n")
    f.write("3. Prophet captured trend/seasonality well (RMSE ~38).\n")
    f.write("4. LSTM achieved the best performance with RMSE ~6, showing the power of deep learning.\n")
    f.write("\nConclusion: LSTM outperformed traditional models for this dataset.\n")

print("Final report generated: final_report.txt")
