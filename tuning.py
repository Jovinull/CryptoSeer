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
import multiprocessing


def get_param_grid():
    return {
        # Tipo de rede
        "model_type": ['LSTM'],

        # Arquitetura da rede
        "units": [32, 64],
        "dropout": [0.0],
        "num_layers": [1],
        "bidirectional": [True],

        # Treinamento
        "batch_size": [16],
        "optimizer": ['adam', 'rmsprop', 'nadam'],
        "learning_rate": [0.01],

        # Entrada de dados
        "prediction_days": [15],
        "future_day": [1],

        # Normalização
        "scaler_type": ['MinMax', 'Standard']
    }

def normalize_metrics(mse, mae, r2, loss):
    # Valores esperados aproximados para normalização
    # Isso pode ser refinado com base em dados históricos reais
    return np.mean([
        mse / 0.01,
        mae / 0.01,
        (1 - r2),
        loss / 0.01
    ])

def evaluate_combination(params):
    import tensorflow as tf
    from model.architecture import build_model
    from tensorflow.keras.callbacks import EarlyStopping
    from data.preprocessing import preprocess_data
    from data.loader import download_data
    from config import crypto_currency, against_currency, start_date, end_date
    import datetime as dt

    try:
        ticker = f"{crypto_currency}-{against_currency}"
        today = dt.datetime.now().strftime("%Y-%m-%d")
        end = end_date or today
        base_data = download_data(ticker, start_date, end)

        x_train, y_train, scaler = preprocess_data(
            base_data.copy(),
            params['prediction_days'],
            params['future_day'],
            scaler_type=params['scaler_type']
        )

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

        return {**params,
                "loss": final_loss,
                "mse": mse,
                "mae": mae,
                "r2": r2,
                "score_final": score_final}

    except Exception as e:
        print(f"⚠️ Erro com combinação: {params} → {e}")
        return None

def hyperparameter_tuning():
    param_grid = get_param_grid()
    keys = list(param_grid.keys())
    combinations = list(itertools.product(*[param_grid[k] for k in keys]))

    print(f"🔧 Total de combinações: {len(combinations)}")

    num_threads = max(1, multiprocessing.cpu_count() // 2)
    print(f"🚀 Executando com {num_threads} processos em paralelo")

    with multiprocessing.Pool(processes=num_threads) as pool:
        results = pool.map(evaluate_combination, [dict(zip(keys, combo)) for combo in combinations])

    results = [r for r in results if r is not None]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(f"results/tuning_results_{timestamp}.csv", index=False)
    print(f"\n✅ Tuning finalizado. {len(results)} combinações salvas em results/tuning_results_{timestamp}.csv")
    return df

if __name__ == "__main__":
    hyperparameter_tuning()
