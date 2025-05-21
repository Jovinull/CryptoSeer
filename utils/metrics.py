import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from datetime import datetime

def evaluate_model(actual, predicted, save_csv=False):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    print(f"\n📊 Avaliação do modelo:")
    print(f" - MSE: {mse:.4f}")
    print(f" - MAE: {mae:.4f}")
    print(f" - R²: {r2:.4f}")

    if save_csv:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs("results", exist_ok=True)
        df = pd.DataFrame([{"timestamp": timestamp, "MSE": mse, "MAE": mae, "R2": r2}])
        df.to_csv(f"results/evaluation_metrics_{timestamp}.csv", index=False)

    return mse, mae, r2
