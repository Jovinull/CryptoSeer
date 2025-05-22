import os
import time
import pandas as pd
import yfinance as yf
import glob


def download_data(ticker: str, start: str, end: str, max_retries=6, cache_dir="data_cache"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")

    # Tenta encontrar cache exato ou algum compatível
    cached_file = cache_file if os.path.exists(cache_file) else find_latest_cache(ticker, cache_dir)

    if cached_file and os.path.exists(cached_file):
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


def find_latest_cache(ticker, cache_dir):
    pattern = os.path.join(cache_dir, f"{ticker}_*.csv")
    files = glob.glob(pattern)
    if files:
        # Ordena pelo nome do arquivo (por data), usa o mais recente
        files.sort(reverse=True)
        return files[0]
    return None