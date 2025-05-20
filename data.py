import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

def download_data(ticker: str, start: str, end: str, max_retries=3):
    for attempt in range(max_retries):
        data = yf.download(ticker, start=start, end=end)
        if not data.empty:
            return data
        wait = 2 ** attempt
        print(f"Tentativa {attempt + 1} falhou. Tentando novamente em {wait} segundos...")
        time.sleep(wait)
    raise ValueError(f"Falha ao baixar dados para {ticker} após {max_retries} tentativas.")


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
