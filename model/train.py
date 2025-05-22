from model.architecture import build_model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd
from datetime import datetime
import os
from config import tuning_batch_size
import joblib

def train_model(x_train, y_train, model_file, model_type, y_scaler=None):
    model = build_model((x_train.shape[1], x_train.shape[2]), model_type)
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    history = model.fit(
        x_train,
        y_train,
        epochs=100,
        batch_size=tuning_batch_size,
        callbacks=callbacks
    )

    model.save(model_file)
    if y_scaler is not None:
        scaler_path = model_file.replace(".h5", "_yscaler.pkl")
        joblib.dump(y_scaler, scaler_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("results", exist_ok=True)
    metrics_df = pd.DataFrame(history.history)
    metrics_df.to_csv(f"results/train_metrics_{timestamp}.csv", index=False)

    return model
