import yfinance as yf
import numpy as np
import pandas as pd
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def download_data(ticker: str, start: str, end: str, max_retries=6):
    for attempt in range(max_retries):
        data = yf.download(ticker, start=start, end=end, progress=False, threads=False)
        if not data.empty:
            return data
        wait = 5 * (attempt + 1)  # espera: 5s, 10s, 15s, ...
        print(f"Tentativa {attempt + 1} falhou. Tentando novamente em {wait} segundos...")
        time.sleep(wait)
    raise ValueError(f"Falha ao baixar dados para {ticker} após {max_retries} tentativas.")

def add_technical_indicators(df):
    df = df.copy()

    # SMA e EMA
    for period in [7, 14, 30]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

    # RSI
    delta = df['Close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.rolling(14).mean()
    ma_down = down.rolling(14).mean()
    rs = ma_up / ma_down
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']

    # Bollinger Bands (20 períodos)
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = sma20 + 2 * std20
    df['BB_lower'] = sma20 - 2 * std20
    df['BB_width'] = df['BB_upper'] - df['BB_lower']

    df.dropna(inplace=True)
    return df


def preprocess_data(data, prediction_days, future_day, scaler=None):
    data = add_technical_indicators(data)

    if scaler is None:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data.values)

    scaled_data = scaler.transform(data.values)

    x_train, y_train = [], []
    for x in range(prediction_days, len(scaled_data) - future_day):
        x_train.append(scaled_data[x - prediction_days:x])
        y_train.append(scaled_data[x + future_day][0])  # preço de fechamento

    x_train, y_train = np.array(x_train), np.array(y_train)

    return x_train, y_train, scaler
