import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def preprocess_data(data, prediction_days, future_day, scaler=None, scaler_type='MinMax'):
    data = add_technical_indicators(data)

    if scaler is None:
        scaler_cls = MinMaxScaler if scaler_type == 'MinMax' else StandardScaler
        scaler = scaler_cls()
        scaler.fit(data.values)

    scaled_data = scaler.transform(data.values)

    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data) - future_day):
        # Verifica se o denominador é próximo de zero
        if abs(scaled_data[x][0]) < 1e-8:
            continue  # ignora essa amostra

        x_train.append(scaled_data[x - prediction_days:x])
        pct_delta = (scaled_data[x + future_day][0] / scaled_data[x][0]) - 1
        y_train.append(pct_delta)

    if not y_train:
        raise ValueError("Nenhum dado válido para y_train. Verifique se o dataset está correto.")

    x_train, y_train = np.array(x_train), np.array(y_train)

    # Normaliza y_train para regressão estável
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

    return x_train, y_train, scaler, y_scaler

def add_technical_indicators(df):
    df = df.copy()

    for period in [7, 14, 30]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / ma_down
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = sma20 + 2 * std20
    df['BB_lower'] = sma20 - 2 * std20
    df['BB_width'] = df['BB_upper'] - df['BB_lower']

    df.dropna(inplace=True)
    return df
