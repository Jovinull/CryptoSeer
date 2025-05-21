# tuning.py (atualizado para testar todos os hiperparâmetros definidos)
import itertools
import pandas as pd
from datetime import datetime
from model.architecture import build_model
from tensorflow.keras.callbacks import EarlyStopping
import os
from data.loader import download_data
from data.preprocessing import preprocess_data
from config import crypto_currency, against_currency, start_date, end_date

import datetime as dt

# Define grid completo de hiperparâmetros
def get_param_grid():
    return {
        "model_type": ['LSTM', 'GRU', 'Dense'],
        "units": [32, 50, 64, 100],
        "dropout": [0.1, 0.2, 0.3],
        "num_layers": [1, 2, 3],
        "bidirectional": [True, False],
        "batch_size": [16, 32, 64],
        "optimizer": ['adam', 'rmsprop'],
        "learning_rate": [0.001, 0.0005, 0.0001],
        "prediction_days": [30, 60],
        "future_day": [1, 5],
        "scaler_type": ['MinMax', 'Standard']
    }

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
            # Override temporário de config para essa combinação
            x_train, y_train, _ = preprocess_data(
                base_data.copy(),
                params['prediction_days'],
                params['future_day'],
                scaler_type=params['scaler_type']
            )

            # Atualiza variáveis globais necessárias para build_model()
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

            results.append({**params, "loss": final_loss})

        except Exception as e:
            print(f"⚠️ Erro com combinação {i+1}: {e}")

    df = pd.DataFrame(results)
    df.to_csv(f"results/tuning_results_{timestamp}.csv", index=False)
    print(f"\n✅ Tuning finalizado. {len(results)} combinações salvas em results/tuning_results_{timestamp}.csv")
    return df

if __name__ == "__main__":
    hyperparameter_tuning()
