# tuning.py (atualizado para usar múltiplas métricas e score_final composto)
import itertools
import pandas as pd
from datetime import datetime
from model.architecture import build_model
from tensorflow.keras.callbacks import EarlyStopping
import os
from data.loader import download_data
from data.preprocessing import preprocess_data
from config import crypto_currency, against_currency, start_date, end_date
from utils.metrics import evaluate_model
import numpy as np
import datetime as dt

def get_param_grid():
    return {
        # Tipo de rede
        "model_type": ['LSTM', 'GRU', 'Dense'],

        # Arquitetura da rede
        "units": [32, 64, 96, 128],
        "dropout": [0.0, 0.1, 0.2, 0.3],
        "num_layers": [1, 2, 3, 4],
        "bidirectional": [True, False],

        # Treinamento
        "batch_size": [16, 32, 64, 128],
        "optimizer": ['adam', 'rmsprop', 'nadam'],
        "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],

        # Entrada de dados
        "prediction_days": [15, 30, 60, 90, 120],
        "future_day": [1, 3, 5, 10, 30],

        # Normalização
        "scaler_type": ['MinMax', 'Standard']
    }


def normalize_metrics(mse, mae, r2, loss):
    # Valores esperados aproximados para normalização
    # Isso pode ser refinado com base em dados históricos reais
    return np.mean([
        mse / 0.01,           # menor é melhor
        mae / 0.01,           # menor é melhor
        (1 - r2),             # menor é melhor
        loss / 0.01           # menor é melhor
    ])

def hyperparameter_tuning():
    param_grid = get_param_grid()
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)

    ticker = f"{crypto_currency}-{against_currency}"
    today = dt.datetime.now().strftime("%Y-%m-%d")
    end = end_date or today
    base_data = download_data(ticker, start_date, end)

    for i, combo in enumerate(combinations):
        params = dict(zip(keys, combo))
        print(f"\n🔍 {i+1}/{len(combinations)} - Testando: {params}")

        try:
            x_train, y_train, scaler = preprocess_data(
                base_data.copy(),
                params['prediction_days'],
                params['future_day'],
                scaler_type=params['scaler_type']
            )

            # Atualiza variáveis globais para arquitetura
            globals().update({
                'tuning_units': params['units'],
                'tuning_dropout': params['dropout'],
                'tuning_num_layers': params['num_layers'],
                'tuning_bidirectional': params['bidirectional'],
                'tuning_optimizer': params['optimizer'],
                'tuning_learning_rate': params['learning_rate']
            })

            model = build_model((x_train.shape[1], x_train.shape[2]), params['model_type'])
            callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
            history = model.fit(x_train, y_train, epochs=50, batch_size=params['batch_size'], callbacks=callbacks, verbose=0)

            final_loss = history.history['loss'][-1]
            predictions = model.predict(x_train, verbose=0).flatten()
            actuals = y_train

            mse = np.mean((actuals - predictions) ** 2)
            mae = np.mean(np.abs(actuals - predictions))
            r2 = 1 - (np.sum((actuals - predictions)**2) / np.sum((actuals - np.mean(actuals))**2))

            score_final = normalize_metrics(mse, mae, r2, final_loss)

            results.append({**params,
                            "loss": final_loss,
                            "mse": mse,
                            "mae": mae,
                            "r2": r2,
                            "score_final": score_final})

        except Exception as e:
            print(f"⚠️ Erro com combinação {i+1}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(f"results/tuning_results_{timestamp}.csv", index=False)
    print(f"\n✅ Tuning finalizado. {len(results)} combinações salvas em results/tuning_results_{timestamp}.csv")
    return df

if __name__ == "__main__":
    hyperparameter_tuning()
