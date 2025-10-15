
'''
Fase 3: Predicción de Top 5 Tickers con Mayor Rentabilidad

Este script carga el modelo GRU de correlación entrenado y los datos enriquecidos
para predecir la rentabilidad futura a 5 días de todos los tickers disponibles.
Finalmente, ordena los resultados y muestra los 5 tickers con la predicción más alta.
'''
import pandas as pd
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
import joblib
from tensorflow.keras.models import load_model

# --- Configuración ---
DATA_PATH = 'all_enriched_data.csv'
MODEL_PATH = 'correlation_model.h5'
SCALER_PATH = 'correlation_scaler.joblib'
SEQUENCE_LENGTH = 30 # Debe ser el mismo que en el entrenamiento

if __name__ == '__main__':
    print("Iniciando Fase 3: Predicción de Top 5 Tickers.")
    start_time = time.time()

    # Cambiar al directorio del script para que las rutas relativas funcionen
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # --- 1. Carga de Modelo y Scaler ---
    if not all(os.path.exists(p) for p in [MODEL_PATH, SCALER_PATH, DATA_PATH]):
        print(f"Error: Faltan archivos necesarios. Asegúrate de que existen:")
        print(f"- Modelo: {MODEL_PATH}")
        print(f"- Scaler: {SCALER_PATH}")
        print(f"- Datos: {DATA_PATH}")
        print("Ejecuta 'data_prep_correlation.py' y 'train_correlation_model.py' primero.")
        exit()

    print(f"Cargando modelo desde '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)

    print(f"Cargando scaler desde '{SCALER_PATH}'...")
    scaler = joblib.load(SCALER_PATH)

    # --- 2. Carga y Preparación de Datos para Predicción ---
    print(f"Cargando datos desde '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)

    feature_columns = [col for col in df.columns if col not in ['ticker', 'future_return_5d']]
    
    print(f"Preparando datos para {df['ticker'].nunique()} tickers...")

    predictions = {}
    grouped = df.groupby('ticker')

    for ticker_name, ticker_data in grouped:
        if len(ticker_data) >= SEQUENCE_LENGTH:
            # Tomar la última secuencia de datos para la predicción
            last_sequence_df = ticker_data.tail(SEQUENCE_LENGTH)
            
            # Extraer solo las features
            sequence_values = last_sequence_df[feature_columns].values
            
            # Escalar la secuencia
            sequence_scaled = scaler.transform(sequence_values)
            
            # Reformatear para el modelo: (1, sequence_length, num_features)
            sequence_reshaped = np.expand_dims(sequence_scaled, axis=0)
            
            # Realizar la predicción
            predicted_return = model.predict(sequence_reshaped, verbose=0)[0][0]
            
            predictions[ticker_name] = predicted_return
        else:
            print(f"Aviso: No hay suficientes datos para el ticker '{ticker_name}' (se requieren {SEQUENCE_LENGTH} días). Se omitirá.")

    # --- 3. Mostrar Resultados ---
    if not predictions:
        print("No se pudo realizar ninguna predicción.")
        exit()

    # Ordenar los tickers por rentabilidad predicha (de mayor a menor)
    sorted_predictions = sorted(predictions.items(), key=lambda item: item[1], reverse=True)

    print("\n--- Predicción de Top 5 Tickers con Mayor Rentabilidad para la Próxima Semana ---")
    for i, (ticker, predicted_return) in enumerate(sorted_predictions[:5]):
        print(f"{i+1}. Ticker: {ticker:<10} | Rentabilidad Predicha a 5 días: {predicted_return:.8f}")

    end_time = time.time()
    print(f"\n¡Fase 3 completada en {end_time - start_time:.2f} segundos!")
