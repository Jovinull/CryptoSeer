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
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def get_param_grid():
    return {
        # Tipo de rede
        "model_type": ['LSTM', 'GRU', 'Dense'],

        # Arquitetura da rede
        "units": [32, 64, 96, 128],              # neurônios por camada
        "dropout": [0.0, 0.1, 0.2, 0.3],          # regularização
        "num_layers": [1, 2, 3],                 # camadas empilhadas
        "bidirectional": [True, False],          # usa camada bidirecional?

        # Treinamento
        "batch_size": [16, 32, 64],              # tamanho do lote
        "optimizer": ['adam', 'rmsprop', 'nadam'],  # otimizadores
        "learning_rate": [0.01, 0.005, 0.001, 0.0005, 0.0001],  # taxa de aprendizado

        # Entrada de dados
        "prediction_days": [15, 30, 60, 90, 120], # histórico usado para prever
        "future_day": [1, 3, 5, 10, 30],          # horizonte de previsão

        # Normalização
        "scaler_type": ['MinMax', 'Standard']    # normalização
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
    try:
        from model.architecture import build_model
        from data.preprocessing import preprocess_data
        from data.loader import download_data
        from config import crypto_currency, against_currency, start_date, end_date
        import datetime as dt

        ticker = f"{crypto_currency}-{against_currency}"
        today = dt.datetime.now().strftime("%Y-%m-%d")
        end = end_date or today
        base_data = download_data(ticker, start_date, end)

        # ⬇️ Usa previsão baseada em delta
        x_train, y_train, scaler, y_scaler = preprocess_data(
            base_data.copy(),
            params['prediction_days'],
            params['future_day'],
            scaler_type=params['scaler_type']
        )

        # Atualiza hiperparâmetros globais
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
        pred_scaled = model.predict(x_train, verbose=0).flatten()
        actual_scaled = y_train

        # ⬇️ Desnormaliza os valores previstos e reais para avaliação coerente
        pred_real = y_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
        actual_real = y_scaler.inverse_transform(actual_scaled.reshape(-1, 1)).flatten()

        mse = np.mean((actual_real - pred_real) ** 2)
        mae = np.mean(np.abs(actual_real - pred_real))
        r2 = 1 - (np.sum((actual_real - pred_real)**2) / np.sum((actual_real - np.mean(actual_real))**2))

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
