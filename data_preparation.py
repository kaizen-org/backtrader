import pandas as pd
import numpy as np
import os
import time

print("Iniciando el script de preparación de datos (VERSIÓN MANUAL)...")
start_time = time.time()

# --- 1.1. Carga de Datos ---
print("\nFase 1.1: Cargando datos...")
data_folder = 'ibkr'
assets = ['A.USD.csv', 'ABM.USD.csv']
all_dfs = []

for asset in assets:
    file_path = os.path.join(data_folder, asset)
    print(f"Cargando {file_path}...")
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['datetime'] = pd.to_datetime(df['datetime'])
        df.set_index('datetime', inplace=True)
        df['ticker'] = asset.split('.')[0]
        all_dfs.append(df)
    else:
        print(f"Warning: File not found at {file_path}")

if not all_dfs:
    print("Error: No data files found. Exiting.")
    exit()

print("Concatenando ficheros...")
full_df = pd.concat(all_dfs)
print(f"Datos cargados. Shape total: {full_df.shape}")

# --- 1.2. Cálculo de Indicadores Técnicos (Manual) ---
print("\nFase 1.2: Calculando indicadores técnicos MANUALMENTE...")

grouped = full_df.groupby('ticker')
processed_dfs = []

for name, group in grouped:
    print(f"Procesando ticker: {name}...")
    df = group.copy()

    # SMAs
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['SMA_200'] = df['close'].rolling(window=200).mean()

    # Bollinger Bands
    df['BBM_20_2.0'] = df['close'].rolling(window=20).mean()
    std_20 = df['close'].rolling(window=20).std()
    df['BBU_20_2.0'] = df['BBM_20_2.0'] + (2 * std_20)
    df['BBL_20_2.0'] = df['BBM_20_2.0'] - (2 * std_20)

    # RSI
    delta = df['close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.ewm(com=14 - 1, adjust=False).mean()
    avg_loss = loss.ewm(com=14 - 1, adjust=False).mean()
    rs = avg_gain / avg_loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df['close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_12_26_9'] = ema_12 - ema_26
    df['MACDs_12_26_9'] = df['MACD_12_26_9'].ewm(span=9, adjust=False).mean()
    df['MACDh_12_26_9'] = df['MACD_12_26_9'] - df['MACDs_12_26_9']
    
    # ATR
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr_df = pd.concat([high_low, high_close, low_close], axis=1)
    tr = tr_df.max(axis=1)
    df['ATRr_14'] = tr.ewm(alpha=1/14, adjust=False).mean()


    # Retornos diarios
    df['daily_return'] = df['close'].pct_change()
    print(f"  Indicadores para {name} calculados.")

    # --- 1.3. Definición de la Variable Objetivo (Target) ---
    print(f"  Definiendo variable objetivo para {name}...")
    FORWARD_WINDOW = 5
    df['future_return'] = df['close'].shift(-FORWARD_WINDOW) / df['close'] - 1
    momentum_profit = np.where(df['SMA_50'] > df['SMA_200'], df['future_return'], -df['future_return'])
    mean_reversion_profit = np.where(df['close'] < df['BBL_20_2.0'], df['future_return'], np.where(df['close'] > df['BBU_20_2.0'], -df['future_return'], 0))
    df['momentum_profit'] = momentum_profit
    df['mean_reversion_profit'] = mean_reversion_profit
    df['estrategia_optima'] = np.where(df['momentum_profit'] > df['mean_reversion_profit'], 1, 0)
    print(f"  Variable objetivo para {name} definida.")

    processed_dfs.append(df)

print("\nCombinando y limpiando datos procesados...")
enriched_df = pd.concat(processed_dfs)
enriched_df.dropna(inplace=True)
print("Filas con NaN eliminadas.")

final_columns = [
    'ticker', 'open', 'high', 'low', 'close', 'volume',
    'SMA_20', 'SMA_50', 'SMA_200', 'RSI_14',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'ATRr_14', 'daily_return', 'estrategia_optima'
]
final_df = enriched_df[final_columns]

output_path = 'enriched_data.csv'
print(f"Guardando DataFrame final en '{output_path}'...")
final_df.to_csv(output_path)

end_time = time.time()
print(f"\n¡Fase 1 completada en {end_time - start_time:.2f} segundos!")
print(f"Datos enriquecidos y guardados en '{output_path}'")
print(f"Dimensiones del DataFrame final: {final_df.shape}")
print("\nPrimeras 5 filas del resultado:")
print(final_df.head())
