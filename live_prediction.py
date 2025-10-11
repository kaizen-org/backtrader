
'''
Fase 4: Predicción para Inversión en Vivo

Este script automatiza el proceso de generar una recomendación de estrategia
(Momentum o Reversión a la Media) para un activo específico para el día siguiente.

Uso:
    python live_prediction.py --ticker A

El script asume que los datos históricos, incluido el del día más reciente,
ya han sido procesados por `data_preparation.py` y están en `enriched_data.csv`.
'''

import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import argparse
import os

# --- Constantes y Configuración ---
MODEL_PATH = 'gru_strategy_selector.h5'
SCALER_PATH = 'gru_scaler.joblib'
DATA_PATH = os.path.join('ibkr', 'output', 'enriched_data.csv')
SEQUENCE_LENGTH = 30

def make_live_prediction(ticker: str):
    '''
    Realiza una predicción de estrategia para un ticker dado para el día siguiente.
    '''
    print(f"Iniciando predicción para el ticker: {ticker}")

    # 1. Cargar datos enriquecidos
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra el archivo de datos '{DATA_PATH}'.")
        print("Asegúrate de haber ejecutado 'data_preparation.py' primero.")
        return

    print(f"Cargando datos desde '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)

    # 2. Filtrar por ticker y obtener la última secuencia
    ticker_df = df[df['ticker'] == ticker]
    if len(ticker_df) < SEQUENCE_LENGTH:
        print(f"Error: No hay suficientes datos para el ticker '{ticker}'. Se necesitan {SEQUENCE_LENGTH} registros, pero solo hay {len(ticker_df)}.")
        return

    print(f"Obteniendo los últimos {SEQUENCE_LENGTH} registros para '{ticker}'...")
    last_sequence_df = ticker_df.tail(SEQUENCE_LENGTH)

    # 3. Seleccionar y escalar las características
    features = [col for col in df.columns if col not in ['ticker', 'estrategia_optima']]
    sequence_data = last_sequence_df[features].values

    print(f"Cargando escalador desde '{SCALER_PATH}'...")
    scaler = joblib.load(SCALER_PATH)

    print("Normalizando la secuencia de datos...")
    scaled_sequence = scaler.transform(sequence_data)
    scaled_sequence = np.expand_dims(scaled_sequence, axis=0) # Añadir dimensión para el batch

    # 4. Cargar modelo y hacer la predicción
    print(f"Cargando modelo desde '{MODEL_PATH}'...")
    model = load_model(MODEL_PATH)

    print("Realizando la predicción...")
    prediction = model.predict(scaled_sequence)
    predicted_value = prediction[0][0]

    # 5. Interpretar y mostrar el resultado
    strategy = "Momentum" if predicted_value > 0.5 else "Reversión a la Media"

    print("\n" + "="*50)
    print(f"RESULTADO DE LA PREDICCIÓN PARA '{ticker}'")
    print(f"Valor predicho por el modelo: {predicted_value:.4f}")
    print(f"Estrategia recomendada para el próximo día de mercado: **{strategy}**")
    print("="*50 + "\n")
    print("Recordatorio: Esta es la recomendación del modelo. Use esta información para aplicar las reglas de entrada/salida de la estrategia correspondiente.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Genera una recomendación de estrategia para un ticker.')
    parser.add_argument('--ticker', type=str, required=True, help='El ticker del activo a predecir (ej. A, ABM).')
    args = parser.parse_args()

    make_live_prediction(args.ticker)
