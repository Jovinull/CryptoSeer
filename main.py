import datetime as dt
import pandas as pd
from data import download_data, preprocess_data
from model import build_model, save_model, load_existing_model
from utils import evaluate_model, plot_predictions, predict_future
import numpy as np

if __name__ == "__main__":
    crypto_currency = 'DOGE'
    against_currency = 'USD'
    start_date = "2014-01-01"
    test_start_date = dt.datetime(2018, 1, 1)
    end_date = dt.datetime.now().strftime("%Y-%m-%d")
    prediction_days = 60
    future_day = 30
    ticker = f"{crypto_currency}-{against_currency}"
    model_file = f"model_{crypto_currency}.h5"

    data = download_data(ticker, start_date, end_date)
    x_train, y_train, scaler = preprocess_data(data, prediction_days, future_day)

    model = load_existing_model(model_file)
    if not model:
        model = build_model((x_train.shape[1], 1))
        model.fit(x_train, y_train, epochs=25, batch_size=32)
        save_model(model, model_file)

    test_data = download_data(ticker, test_start_date.strftime("%Y-%m-%d"), end_date)
    actual_prices = test_data['Close'].values
    total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    x_test = []
    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x - prediction_days:x, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    prediction_prices = model.predict(x_test)
    prediction_prices = scaler.inverse_transform(prediction_prices)

    evaluate_model(actual_prices, prediction_prices)
    plot_predictions(test_data.index, actual_prices, prediction_prices, crypto_currency)

    future_price = predict_future(model, model_inputs, scaler, prediction_days)
    print(f"\nPrevisão de preço para os próximos {future_day} dias: ${future_price:.2f}")
