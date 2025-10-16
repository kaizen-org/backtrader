import pandas as pd
import numpy as np
import os
import glob
import time
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor

# --- Configuración ---
FORWARD_WINDOW = 5  # Predecir la rentabilidad a 5 días
N_PEERS = 5         # Número de pares a considerar para cada ticker

def normalize_ticker(ticker):
    '''Limpia y normaliza el nombre de un ticker.'''
    ticker = re.sub(r'\.USD$', '', ticker, flags=re.IGNORECASE)
    ticker = re.sub(r'[ .]+', '_', ticker)
    ticker = ticker.rstrip('_')
    return ticker

def _calculate_rsi(series, period=14):
    delta = series.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=period - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=period - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def process_chunk(df_chunk, peer_map, market_returns, returns_df):
    """
    Enriquece un trozo del DataFrame (un ticker completo) con indicadores y datos de pares.
    """
    df_chunk['SMA_20'] = df_chunk['close'].rolling(window=20).mean()
    df_chunk['SMA_50'] = df_chunk['close'].rolling(window=50).mean()
    df_chunk['RSI_14'] = _calculate_rsi(df_chunk['close'], 14)
    bb_window = 20
    std_20 = df_chunk['close'].rolling(window=bb_window).std()
    df_chunk['BBM_20_2.0'] = df_chunk['close'].rolling(window=bb_window).mean()
    df_chunk['BBU_20_2.0'] = df_chunk['BBM_20_2.0'] + (2 * std_20)
    df_chunk['BBL_20_2.0'] = df_chunk['BBM_20_2.0'] - (2 * std_20)
    ema_12 = df_chunk['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df_chunk['close'].ewm(span=26, adjust=False).mean()
    df_chunk['MACD_12_26_9'] = ema_12 - ema_26
    df_chunk['MACDs_12_26_9'] = df_chunk['MACD_12_26_9'].ewm(span=9, adjust=False).mean()

    df_chunk['daily_return'] = df_chunk['close'].pct_change()
    df_chunk['future_return_5d'] = (df_chunk['close'].shift(-FORWARD_WINDOW) / df_chunk['close']) - 1

    df_chunk = df_chunk.join(market_returns, how='left')

    ticker_name = df_chunk['ticker'].iloc[0]
    df_chunk['peer_avg_return'] = df_chunk.apply(
        lambda row: returns_df.loc[row.name, peer_map.get(str(row.name.year), {}).get(ticker_name, [])].mean(),
        axis=1
    )
    
    return df_chunk

if __name__ == '__main__':
    print("Iniciando Fase 1 (Revisada): Preparación de datos con contexto de correlación.")
    start_time = time.time()

    IN_COLAB = 'google.colab' in sys.modules

    if IN_COLAB:
        print("Entorno de Google Colab detectado. Montando Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive')
        
        BASE_DIR = '/content/drive/MyDrive/backtrader_colab'
        os.makedirs(BASE_DIR, exist_ok=True)
        DATA_FOLDER = os.path.join(BASE_DIR, 'ibkr', 'historic') 
        
        OUTPUT_DATA_PATH = os.path.join(BASE_DIR, 'all_enriched_data.csv')
        OUTPUT_PEERS_PATH = os.path.join(BASE_DIR, 'peer_groups.json')

        print(f"Ruta de datos de entrada (Drive): {DATA_FOLDER}")
        print(f"Rutas de datos de salida (Drive): {BASE_DIR}")
    else:
        print("Entorno local detectado.")
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        DATA_FOLDER = os.path.join('..', 'historic') 
        OUTPUT_DATA_PATH = 'all_enriched_data.csv'
        OUTPUT_PEERS_PATH = 'peer_groups.json'

    all_files = glob.glob(os.path.join(DATA_FOLDER, '*.csv'))
    print(f"Encontrados {len(all_files)} ficheros CSV.")

    all_dfs = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, usecols=['datetime', 'close'])
            if not df.empty:
                ticker_raw = os.path.basename(file_path).removesuffix('.csv')
                ticker = normalize_ticker(ticker_raw)
                df.rename(columns={'close': ticker}, inplace=True)
                df['datetime'] = pd.to_datetime(df['datetime'])
                df.set_index('datetime', inplace=True)
                all_dfs.append(df)
        except Exception as e:
            print(f"Warning: No se pudo cargar {file_path}: {e}")

    print("Combinando rentabilidades de todos los tickers...")
    returns_df = pd.concat(all_dfs, axis=1)
    returns_df = returns_df.groupby(level=0, axis=1).mean()
    returns_df = returns_df.pct_change().fillna(0)

    print("Calculando grupos de pares anuales...")
    years = returns_df.index.year.unique()
    peer_map = {}
    for year in years:
        peer_map[str(year)] = {}
        corr_matrix = returns_df[returns_df.index.year == year].corr()
        
        for ticker in corr_matrix.columns:
            top_peers = corr_matrix[ticker].drop(ticker).sort_values(ascending=False).head(N_PEERS).index.tolist()
            peer_map[str(year)][ticker] = top_peers

    print(f"Guardando mapa de pares en '{OUTPUT_PEERS_PATH}'...")
    with open(OUTPUT_PEERS_PATH, 'w') as f:
        json.dump(peer_map, f, indent=4)

    print("Enriqueciendo datos con indicadores y features de contexto...")
    market_returns = returns_df.mean(axis=1).rename('market_avg_return')
    
    full_data_list = []
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path)
            ticker_raw = os.path.basename(file_path).removesuffix('.csv')
            df['ticker'] = normalize_ticker(ticker_raw)
            full_data_list.append(df)
        except Exception as e:
            print(f"Warning: No se pudo cargar {file_path} para el enriquecimiento: {e}")

    full_raw_df = pd.concat(full_data_list)
    full_raw_df['datetime'] = pd.to_datetime(full_raw_df['datetime'])
    full_raw_df.set_index('datetime', inplace=True)

    grouped = full_raw_df.groupby('ticker')
    ticker_chunks = [group for name, group in grouped]

    enriched_dfs = []
    with ProcessPoolExecutor() as executor:
        results = [executor.submit(process_chunk, chunk, peer_map, market_returns, returns_df) for chunk in ticker_chunks]
        for future in results:
            res = future.result()
            if res is not None:
                enriched_dfs.append(res)

    print(f"\nCombinando datos enriquecidos de {len(enriched_dfs)} tickers...")
    final_df = pd.concat(enriched_dfs)

    print("Limpiando datos finales (eliminando filas con valores NaN)...")
    original_rows = len(final_df)
    final_df.dropna(inplace=True)
    cleaned_rows = len(final_df)
    print(f"Se eliminaron {original_rows - cleaned_rows} filas.")

    print(f"Guardando DataFrame final en '{OUTPUT_DATA_PATH}'...")
    final_df.to_csv(OUTPUT_DATA_PATH)

    end_time = time.time()
    print(f"\n¡Fase 1 (Revisada) completada en {end_time - start_time:.2f} segundos!")
    print(f"Datos guardados en '{OUTPUT_DATA_PATH}' con {len(final_df)} filas.")