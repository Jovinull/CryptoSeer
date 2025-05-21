import datetime as dt
from config import *
from data.loader import download_data
from data.preprocessing import preprocess_data
from model.train import train_model
from model.architecture import load_existing_model
from utils.metrics import evaluate_model
from utils.visualization import plot_predictions
from utils.visualization import predict_future, recursive_forecast
import pandas as pd
import numpy as np

if __name__ == "__main__":
    ticker = f"{crypto_currency}-{against_currency}"
    today = dt.datetime.now().strftime("%Y-%m-%d")
    end = end_date or today

    data = download_data(ticker, start_date, end)
    x_train, y_train, scaler = preprocess_data(data, prediction_days, future_day)

    model = load_existing_model(model_file)
    if not model:
        model = train_model(x_train, y_train, model_file, model_type)

    test_data = download_data(ticker, test_start_date, end)
    test_data.index = pd.to_datetime(test_data.index, errors="coerce")
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data, test_data), axis=0)

    from data.preprocessing import add_technical_indicators
    total_dataset = add_technical_indicators(total_dataset)

    start_idx = len(total_dataset) - len(test_data) - prediction_days
    if start_idx < 0:
        raise ValueError("Dados insuficientes para previsão. Ajuste as datas ou aumente o dataset.")
    model_inputs = total_dataset[start_idx:].values
    model_inputs = scaler.transform(model_inputs)

    x_test = [model_inputs[x - prediction_days:x] for x in range(prediction_days, len(model_inputs))]
    x_test = np.array(x_test)

    prediction_prices = model.predict(x_test)
    full_pred = np.zeros((prediction_prices.shape[0], scaler.scale_.shape[0]))
    full_pred[:, 0] = prediction_prices[:, 0]
    prediction_prices = scaler.inverse_transform(full_pred)[:, 0]

    evaluate_model(actual_prices, prediction_prices)
    plot_predictions(test_data.index, actual_prices, prediction_prices, crypto_currency)

    future_price = predict_future(model, model_inputs, scaler, prediction_days)
    print(f"\nPrevisão de preço para os próximos {future_day} dias: ${future_price:.2f}")

    forecast = recursive_forecast(model, model_inputs[-prediction_days:], forecast_horizon, scaler)
    print(f"\nPrevisão recursiva para os próximos {forecast_horizon} dias:")
    for i, price in enumerate(forecast, 1):
        print(f"Dia +{i}: ${price[0]:.2f}")
        
    # Baseline ingênuo
    naive_pred = actual_prices[:-1]
    naive_actual = actual_prices[1:]
    from utils.metrics import evaluate_model
    print("\n📉 Baseline Ingênuo (shift(1)):")
    evaluate_model(naive_actual, naive_pred)
