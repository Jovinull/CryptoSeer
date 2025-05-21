# config.py
import os
import glob
import pandas as pd

# parâmetros gerais
crypto_currency = 'DOGE'
against_currency = 'USD'
start_date = "2020-01-01"
test_start_date = "2022-01-01"
end_date = None
prediction_days = 60
future_day = 30
forecast_horizon = 15
model_file = f"model_{crypto_currency}.h5"
model_type = 'LSTM'  # 'GRU' ou 'Dense' também são válidos

# Carrega melhor tuning automaticamente do CSV mais recente
def load_best_tuning():
    result_files = glob.glob("results/tuning_results_*.csv")
    if not result_files:
        return None
    latest = max(result_files, key=os.path.getctime)
    df = pd.read_csv(latest)
    best = df.sort_values(by='loss').iloc[0]
    return {
        "units": int(best["units"]),
        "dropout": float(best["dropout"]),
        "batch_size": int(best["batch_size"])
    }

best_params = load_best_tuning()
if best_params:
    tuning_units = best_params["units"]
    tuning_dropout = best_params["dropout"]
    tuning_batch_size = best_params["batch_size"]
else:
    tuning_units = 50
    tuning_dropout = 0.2
    tuning_batch_size = 32
