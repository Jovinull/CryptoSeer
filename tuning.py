import itertools
import pandas as pd
from datetime import datetime
from model.architecture import build_model
from tensorflow.keras.callbacks import EarlyStopping
import os

def hyperparameter_tuning(x_train, y_train, param_grid):
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)

    combinations = list(itertools.product(
        param_grid['units'], param_grid['dropout'], param_grid['batch_size']
    ))

    for i, (units, dropout, batch_size) in enumerate(combinations):
        print(f"🔍 Testando combinação {i+1}/{len(combinations)}: units={units}, dropout={dropout}, batch_size={batch_size}")
        model = build_model((x_train.shape[1], x_train.shape[2]))
        model.layers[0].units = units
        model.layers[1].rate = dropout

        callbacks = [EarlyStopping(patience=5, restore_best_weights=True)]
        history = model.fit(
            x_train, y_train, 
            epochs=50, 
            batch_size=batch_size, 
            callbacks=callbacks, 
            verbose=0
        )

        final_loss = history.history['loss'][-1]
        results.append({"units": units, "dropout": dropout, "batch_size": batch_size, "loss": final_loss})

    df = pd.DataFrame(results)
    df.to_csv(f"results/tuning_results_{timestamp}.csv", index=False)
    print("✅ Tuning finalizado. Resultados salvos em 'results/tuning_results_*.csv'")
    return df

if __name__ == "__main__":
    import numpy as np
    from data.loader import download_data
    from data.preprocessing import preprocess_data
    from config import crypto_currency, against_currency, start_date, end_date, prediction_days, future_day

    ticker = f"{crypto_currency}-{against_currency}"
    import datetime as dt
    today = dt.datetime.now().strftime("%Y-%m-%d")
    end = end_date or today

    data = download_data(ticker, start_date, end)
    x_train, y_train, _ = preprocess_data(data, prediction_days, future_day)

    param_grid = {
        "units": [32, 50, 64],
        "dropout": [0.1, 0.2, 0.3],
        "batch_size": [16, 32, 64]
    }

    hyperparameter_tuning(x_train, y_train, param_grid)
