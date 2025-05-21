# config.py (atualizado para usar score_final como critério principal)
import os
import glob
import pandas as pd

# parâmetros fixos
crypto_currency = 'DOGE'
against_currency = 'USD'
start_date = "2020-01-01"
test_start_date = "2022-01-01"
end_date = None
model_file = f"model_{crypto_currency}.h5"

# Carrega melhor tuning automaticamente do CSV mais recente (com base em score_final)
def load_best_tuning():
    result_files = glob.glob("results/tuning_results_*.csv")
    if not result_files:
        return None
    latest = max(result_files, key=os.path.getctime)
    df = pd.read_csv(latest)
    # Usa score_final como critério principal (se existir)
    if 'score_final' in df.columns:
        best = df.sort_values(by='score_final').iloc[0]
    else:
        best = df.sort_values(by='loss').iloc[0]

    return {
        "model_type": best["model_type"],
        "units": int(best["units"]),
        "dropout": float(best["dropout"]),
        "batch_size": int(best["batch_size"]),
        "optimizer": best["optimizer"],
        "learning_rate": float(best["learning_rate"]),
        "num_layers": int(best["num_layers"]),
        "bidirectional": bool(best["bidirectional"]),
        "prediction_days": int(best["prediction_days"]),
        "future_day": int(best["future_day"]),
        "scaler_type": best["scaler_type"]
    }

best_params = load_best_tuning()
if best_params:
    model_type = best_params["model_type"]
    tuning_units = best_params["units"]
    tuning_dropout = best_params["dropout"]
    tuning_batch_size = best_params["batch_size"]
    tuning_optimizer = best_params["optimizer"]
    tuning_learning_rate = best_params["learning_rate"]
    tuning_num_layers = best_params["num_layers"]
    tuning_bidirectional = best_params["bidirectional"]
    prediction_days = best_params["prediction_days"]
    future_day = best_params["future_day"]
    scaler_type = best_params["scaler_type"]
else:
    model_type = 'LSTM'
    tuning_units = 50
    tuning_dropout = 0.2
    tuning_batch_size = 32
    tuning_optimizer = 'adam'
    tuning_learning_rate = 0.001
    tuning_num_layers = 2
    tuning_bidirectional = True
    prediction_days = 60
    future_day = 30
    scaler_type = 'MinMax'

forecast_horizon = 15
