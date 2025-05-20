import yfinance as yf
import numpy as np
import pandas as pd
import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import glob


def find_latest_cache(ticker, cache_dir):
    pattern = os.path.join(cache_dir, f"{ticker}_*.csv")
    files = glob.glob(pattern)
    if files:
        # Ordena pelo nome do arquivo (por data), usa o mais recente
        files.sort(reverse=True)
        return files[0]
    return None


def download_data(ticker: str, start: str, end: str, max_retries=6, cache_dir="data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")

    # Tenta encontrar cache exato ou algum compatível
    cached_file = cache_file if os.path.exists(cache_file) else find_latest_cache(ticker, cache_dir)

    if cached_file and os.path.exists(cached_file):
        print(f"🔄 Usando cache local: {cached_file}")
        try:
            df = pd.read_csv(cached_file, index_col=0, parse_dates=True)

            # Força tipos numéricos
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df.dropna(inplace=True)

            if df.empty:
                raise ValueError("Arquivo de cache está vazio após limpeza.")

            return df

        except Exception as e:
            print(f"⚠️ Erro ao usar cache ({e}). Recriando...")
            os.remove(cached_file)

    # Se não houver cache válido, baixa da API
    for attempt in range(max_retries):
        print(f"⬇️  Buscando dados da API para {ticker} ({start} a {end}) - tentativa {attempt + 1}...")
        try:
            data = yf.download(ticker, start=start, end=end, progress=False, threads=False)
            if not data.empty:
                data.to_csv(cache_file)
                print(f"✅ Dados salvos em cache local: {cache_file}")
                return data
        except Exception as e:
            print(f"⚠️ Erro ao acessar API: {e}")

        wait = 5 * (attempt + 1)
        print(f"⚠️ Tentativa {attempt + 1} falhou. Tentando novamente em {wait} segundos...")
        time.sleep(wait)

    raise ValueError(f"❌ Falha ao baixar dados para {ticker} após {max_retries} tentativas.")


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Médias móveis simples e exponenciais
    for period in [7, 14, 30, 50, 100]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()

    # Bollinger Bands
    sma20 = df['Close'].rolling(window=20).mean()
    std20 = df['Close'].rolling(window=20).std()
    df['BB_upper'] = sma20 + 2 * std20
    df['BB_lower'] = sma20 - 2 * std20
    df['BB_width'] = df['BB_upper'] - df['BB_lower']
    df['BB_percent'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])

    # RSI (Relative Strength Index)
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

    # ROC (Rate of Change)
    df['ROC'] = df['Close'].pct_change(periods=12) * 100

    # Stochastic Oscillator
    low14 = df['Low'].rolling(window=14).min()
    high14 = df['High'].rolling(window=14).max()
    df['Stoch_%K'] = 100 * ((df['Close'] - low14) / (high14 - low14))
    df['Stoch_%D'] = df['Stoch_%K'].rolling(window=3).mean()

    # Williams %R
    df['Williams_%R'] = -100 * ((high14 - df['Close']) / (high14 - low14))

    # OBV (On-Balance Volume)
    obv = [0]
    for i in range(1, len(df)):
        if df['Close'].iloc[i] > df['Close'].iloc[i - 1]:
            obv.append(obv[-1] + df['Volume'].iloc[i])
        elif df['Close'].iloc[i] < df['Close'].iloc[i - 1]:
            obv.append(obv[-1] - df['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    df['OBV'] = obv

    # CMO (Chande Momentum Oscillator)
    cmo_up = up.rolling(14).sum()
    cmo_down = down.rolling(14).sum()
    df['CMO'] = 100 * (cmo_up - cmo_down) / (cmo_up + cmo_down)

    # Limpa NaNs gerados pelas janelas
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
