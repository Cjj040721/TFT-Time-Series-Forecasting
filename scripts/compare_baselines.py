"""
Author     : Cao Junjia

compare_baselines.py — Compare TFT, LSTM, and Informer model performance metrics.
Loads metrics.json from result directories and prints a comparison table.
"""
import json
import pandas as pd

models = ['tft', 'lstm', 'informer']
results = {}

for model in models:
    path = f"results/ETTh1/{model}/seq96_pred24_run1/metrics.json"
    try:
        with open(path, 'r') as f:
            data = json.load(f)
        results[model] = {
            'MAE': round(data['mae'], 4),
            'MSE': round(data['mse'], 4),
            'RMSE': round(data['rmse'], 4),
            'MAPE': round(data['mape'], 4)
        }
    except FileNotFoundError:
        results[model] = {'error': 'metrics.json not found'}

df = pd.DataFrame(results).T
print("Model Comparison (ETTh1  seq=96 -> pred=24)")
print(df)
df.to_csv('results/metrics/baselines_comparison.csv', float_format='%.4f')
print("\nTable saved to results/metrics/baselines_comparison.csv")
