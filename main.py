import datetime as dt
from config import *
from data.loader import download_data
from data.preprocessing import preprocess_data, add_technical_indicators
from model.train import train_model
from model.architecture import load_existing_model
from utils.metrics import evaluate_model
from utils.visualization import plot_predictions, predict_future, recursive_forecast
import pandas as pd
import numpy as np
import warnings
import os
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    ticker = f"{crypto_currency}-{against_currency}"
    today = dt.datetime.now().strftime("%Y-%m-%d")
    end = end_date or today

    data = download_data(ticker, start_date, end)
    x_train, y_train, scaler, y_scaler = preprocess_data(data, prediction_days, future_day)

    model = load_existing_model(model_file)
    if not model:
        model = train_model(x_train, y_train, model_file, model_type, y_scaler)

    # Tenta carregar o y_scaler salvo
    scaler_path = model_file.replace(".h5", "_yscaler.pkl")
    if os.path.exists(scaler_path):
        y_scaler = joblib.load(scaler_path)
    else:
        raise ValueError("y_scaler não encontrado. Treine novamente com o novo formato.")

    test_data = download_data(ticker, test_start_date, end)
    test_data.index = pd.to_datetime(test_data.index, errors="coerce")
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data, test_data), axis=0)
    total_dataset = add_technical_indicators(total_dataset)

    start_idx = len(total_dataset) - len(test_data) - prediction_days
    if start_idx < 0:
        raise ValueError("Dados insuficientes para previsão. Ajuste as datas ou aumente o dataset.")
    model_inputs = total_dataset[start_idx:].values
    model_inputs = scaler.transform(model_inputs)

    x_test = [model_inputs[x - prediction_days:x] for x in range(prediction_days, len(model_inputs))]
    x_test = np.array(x_test)

    # Faz predições e reverte o delta percentual para preço real
    prediction_deltas = model.predict(x_test).flatten()
    prediction_deltas = y_scaler.inverse_transform(prediction_deltas.reshape(-1, 1)).flatten()

    base_prices = actual_prices[:len(prediction_deltas)]
    prediction_prices = base_prices * (1 + prediction_deltas)

    evaluate_model(actual_prices[:len(prediction_prices)], prediction_prices)
    plot_predictions(test_data.index[-len(prediction_prices):], actual_prices[:len(prediction_prices)], prediction_prices, crypto_currency)

    # Previsão de futuro (ajuste necessário para variação percentual também)
    future_price = predict_future(model, model_inputs, scaler, prediction_days, y_scaler, actual_prices[-1])
    print(f"\nPrevisão de preço para os próximos {future_day} dias: ${future_price:.2f}")

    forecast = recursive_forecast(model, model_inputs[-prediction_days:], forecast_horizon, scaler, y_scaler, actual_prices[-1])
    print(f"\nPrevisão recursiva para os próximos {forecast_horizon} dias:")
    for i, price in enumerate(forecast, 1):
        print(f"Dia +{i}: ${price[0]:.2f}")

    naive_pred = actual_prices[:-1]
    naive_actual = actual_prices[1:]
    print("\n📉 Baseline Ingênuo (shift(1)):")
    evaluate_model(naive_actual, naive_pred)
