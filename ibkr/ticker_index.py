import pandas as pd
import yfinance as yf
import urllib.request

# URL del archivo que contiene los tickers de NYSE, AMEX, etc.
url = "ftp://ftp.nasdaqtrader.com/SymbolDirectory/otherlisted.txt"
local_filename_txt = "otherlisted.txt"
local_filename_csv = "nyse_tickers.csv"

# Descargar el archivo de texto
print(f"Descargando archivo desde {url}...")
urllib.request.urlretrieve(url, local_filename_txt)
print(f"Archivo guardado como '{local_filename_txt}'")

# Leer los datos usando pandas. '|' es el separador en este archivo.
# La primera fila es la cabecera (header=0).
df = pd.read_csv(local_filename_txt, sep='|')

# Filtrar el DataFrame para obtener solo los tickers de la NYSE
# La columna 'Exchange' nos dice a qué bolsa pertenece cada ticker.
nyse_tickers_df = df[df['Exchange'] == 'N']

# Guardar el DataFrame filtrado en un archivo CSV
nyse_tickers_df.to_csv(local_filename_csv, index=False)
print(f"Tickers de la NYSE guardados en '{local_filename_csv}'")

# Convertir la columna de símbolos a una lista de Python
nyse_ticker_list = nyse_tickers_df['ACT Symbol'].tolist()

print(f"Se encontraron {len(nyse_ticker_list)} tickers en la NYSE.")
print("Ejemplo de los primeros 10 tickers:", nyse_ticker_list[:10])

# --- Ahora puedes usar esta lista con yfinance ---

# Ejemplo: Obtener la información del precio de cierre de los primeros 5 tickers
if nyse_ticker_list:
    primeros_cinco = nyse_ticker_list[:5]
    data = yf.download(primeros_cinco, period='1d')['Adj Close']
    
    print("\nPrecio de cierre de los 5 primeros tickers de la lista:")
    print(data)