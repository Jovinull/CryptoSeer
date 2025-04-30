import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def download_data(ticker: str, start: str, end: str):
    return yf.download(ticker, start=start, end=end)


def preprocess_data(data, prediction_days, future_day, scaler=None):
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data['Close'].values.reshape(-1, 1))

    scaled_data = scaler.transform(data['Close'].values.reshape(-1, 1))

    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x - prediction_days:x, 0])
        y_train.append(scaled_data[x + future_day, 0])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    return x_train, y_train, scaler
