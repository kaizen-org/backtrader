
'''
Script para descargar datos históricos de Yahoo Finance y guardarlos en formato Parquet.
'''
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def download_data():
    '''Descarga y guarda los datos para GOLD.PA.'''
    ticker_symbol = 'GOLD.PA'
    output_file = f'{ticker_symbol}.parquet'
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)

    print(f"Descargando datos para {ticker_symbol} desde {start_date.strftime('%Y-%m-%d')} hasta {end_date.strftime('%Y-%m-%d')}")
    
    # Descargar los datos
    data = yf.download(ticker_symbol, start=start_date, end=end_date)

    if data.empty:
        print(f"No se encontraron datos para {ticker_symbol}. El ticker podría ser incorrecto o no tener datos en el rango de fechas.")
        return

    # Guardar en formato Parquet
    data.to_parquet(output_file)
    print(f'\nDatos guardados exitosamente en: {output_file}')

def verify_data():
    '''Lee el archivo Parquet y muestra las primeras filas para verificar.'''
    ticker_symbol = 'GOLD.PA'
    input_file = f'{ticker_symbol}.parquet'
    
    try:
        df = pd.read_parquet(input_file)
        print(f'\n--- Verificación de {input_file} ---')
        print('Datos cargados desde el archivo Parquet:')
        print(df.head())
        print('\nÚltimas filas del archivo:')
        print(df.tail())
        print(f"\nEl archivo contiene {len(df)} filas de datos.")
        print('--- Fin de la verificación ---')
    except FileNotFoundError:
        print(f"Error: No se encontró el archivo {input_file}. Asegúrate de que la descarga fue exitosa.")

if __name__ == "__main__":
    # Instalar dependencias primero
    try:
        import yfinance
        import pandas
        import pyarrow
    except ImportError:
        print("Instalando dependencias (yfinance, pandas, pyarrow)...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance", "pandas", "pyarrow"])

    download_data()
    verify_data()
