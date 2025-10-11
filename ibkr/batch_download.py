import pandas as pd
import subprocess
import os
import sys

def run_batch_download():
    tickers_csv_path = 'C:\\Users\\gabriel.casas\\backtrader\\nyse_tickers.csv'
    
    try:
        tickers_df = pd.read_csv(tickers_csv_path, usecols=['ACT Symbol'])
    except FileNotFoundError:
        print(f"Error: El archivo {tickers_csv_path} no fue encontrado.")
        return
    except ValueError as e:
        print(f"Error: No se pudo encontrar la columna 'ACT Symbol' en {tickers_csv_path}. Error: {e}")
        return

    tickers = tickers_df['ACT Symbol'].dropna().unique().tolist()
    
    download_script_path = 'C:\\Users\\gabriel.casas\\backtrader\\ibkr\\download_ibkr.py'
    script_dir = 'C:\\Users\\gabriel.casas\\backtrader\\ibkr'

    print(f"Se encontraron {len(tickers)} tickers únicos para descargar.")

    for i, ticker in enumerate(tickers):
        if '$' in ticker:
            print(f"({i+1}/{len(tickers)}) Omitiendo ticker de acciones preferentes: {ticker}")
            continue

        symbol_for_ibkr = ticker.replace('.', ' ')

        print(f"--- ({i+1}/{len(tickers)}) Descargando {ticker} (usando: {symbol_for_ibkr}) ---")
        
        command = [
            sys.executable,
            download_script_path,
            '--symbol', symbol_for_ibkr,
            '--sectype', 'STK',
            '--exchange', 'SMART',
            '--currency', 'USD',
            '--duration', '5 Y',
            '--timeframe', '1 day'
        ]
        
        try:
            subprocess.run(command, check=True, cwd=script_dir, timeout=300)
        except subprocess.CalledProcessError as e:
            print(f"Error descargando {ticker}: {e}")
        except subprocess.TimeoutExpired:
            print(f"Timeout descargando {ticker}. El proceso tardó más de 5 minutos.")
        except Exception as e:
            print(f"Ocurrió un error inesperado descargando {ticker}: {e}")

if __name__ == '__main__':
    run_batch_download()
