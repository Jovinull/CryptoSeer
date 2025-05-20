import datetime as dt
import pandas as pd
from data import add_technical_indicators, download_data, preprocess_data
from model import build_model, save_model, load_existing_model
from utils import evaluate_model, plot_predictions, predict_future, recursive_forecast
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

if __name__ == "__main__":
    crypto_currency = 'DOGE'
    against_currency = 'USD'
    start_date = "2020-01-01"
    test_start_date = dt.datetime(2022, 1, 1)
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    prediction_days = 60
    future_day = 30
    forecast_horizon = 15  # quantidade de dias futuros para prever recursivamente
    ticker = f"{crypto_currency}-{against_currency}"
    model_file = f"model_{crypto_currency}.h5"

    data = download_data(ticker, start_date, end_date)
    x_train, y_train, scaler = preprocess_data(data, prediction_days, future_day)

    model = load_existing_model(model_file)
    if not model:
        model = build_model((x_train.shape[1], x_train.shape[2]))

        callbacks = [
            EarlyStopping(patience=5, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=3)
        ]

        model.fit(x_train, y_train, epochs=100, batch_size=32, callbacks=callbacks)
        save_model(model, model_file)

    test_data = download_data(ticker, test_start_date.strftime("%Y-%m-%d"), end_date)
    test_data.index = pd.to_datetime(test_data.index, errors="coerce")
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data, test_data), axis=0)
    total_dataset = add_technical_indicators(total_dataset)

    # Reaplique o mesmo scaler
    start_idx = len(total_dataset) - len(test_data) - prediction_days
    if start_idx < 0:
        raise ValueError("Intervalo de dados insuficiente para fazer a previsão. Verifique as datas ou aumente o dataset.")
    model_inputs = total_dataset[start_idx:].values
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x])
    x_test = np.array(x_test)

    prediction_prices = model.predict(x_test)
    # Cria um array com zeros com o mesmo número de colunas do scaler
    full_pred = np.zeros((prediction_prices.shape[0], scaler.scale_.shape[0]))
    full_pred[:, 0] = prediction_prices[:, 0]  # coloca as previsões na coluna do 'Close'
    prediction_prices = scaler.inverse_transform(full_pred)[:, 0]  # pega só a coluna do 'Close'

    evaluate_model(actual_prices, prediction_prices)
    plot_predictions(test_data.index, actual_prices, prediction_prices, crypto_currency)

    future_price = predict_future(model, model_inputs, scaler, prediction_days)
    print(f"\nPrevisão de preço para os próximos {future_day} dias: ${future_price:.2f}")

    # Previsão recursiva para múltiplos dias no futuro
    forecast = recursive_forecast(model, model_inputs[-prediction_days:], forecast_horizon, scaler)
    print(f"\nPrevisão recursiva para os próximos {forecast_horizon} dias:")
    for i, price in enumerate(forecast, 1):
        print(f"Dia +{i}: ${price[0]:.2f}")
