
'''
Pre-cálculo de Predicciones del Modelo

Este script carga el modelo GRU y los datos, y pre-calcula las decisiones
(Momentum vs. Reversión) para el conjunto de datos de validación (out-of-sample).
El resultado se guarda en predictions.csv para ser consumido por el backtester.
'''

import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
import time

print("Iniciando pre-cálculo de predicciones...")
start_time = time.time()

# --- Constantes y Carga de Archivos ---
SEQUENCE_LENGTH = 30
MODEL_PATH = 'gru_strategy_selector.h5'
SCALER_PATH = 'gru_scaler.joblib'
DATA_ENRICHED_PATH = 'enriched_data.csv'

# Cargar modelo, scaler y datos
model = load_model(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
df = pd.read_csv(DATA_ENRICHED_PATH, index_col='datetime', parse_dates=True)

# --- Preparación de Datos (similar al training) ---

# Seleccionar las mismas features que en el entrenamiento
feature_columns = [
    'open', 'high', 'low', 'close', 'volume',
    'SMA_20', 'SMA_50', 'SMA_200',
    'RSI_14',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'ATRr_14',
    'daily_return'
]

df_features = df[feature_columns]

# Dividir en train/test para encontrar el punto de corte exacto
split_ratio = 0.8
split_index = int(len(df_features) * split_ratio)

# Nos quedamos solo con la parte de validación (out-of-sample)
validation_features_raw = df_features[split_index:]

print(f"Procesando {len(validation_features_raw)} filas del conjunto de validación.")

# Escalar los datos de validación
validation_features_scaled = scaler.transform(validation_features_raw)

# --- Creación de Secuencias y Predicción ---

def create_sequences_for_prediction(features, seq_length):
    X_seq = []
    for i in range(len(features) - seq_length + 1):
        X_seq.append(features[i:i+seq_length])
    return np.array(X_seq)

X_pred_seq = create_sequences_for_prediction(validation_features_scaled, SEQUENCE_LENGTH)

print(f"Generadas {len(X_pred_seq)} secuencias para predicción.")

# Predecir todas las secuencias de una sola vez
predictions_proba = model.predict(X_pred_seq, verbose=1)

# Convertir probabilidades a decisiones (0 o 1)
predictions = (predictions_proba > 0.5).astype(int).flatten()

# --- Guardar Predicciones ---

# Las predicciones corresponden a las fechas a partir del final de la primera secuencia
prediction_dates = validation_features_raw.index[SEQUENCE_LENGTH-1:]

# Crear un DataFrame con las fechas y sus predicciones
predictions_df = pd.DataFrame(
    data={'predicted_strategy': predictions},
    index=prediction_dates
)

# Necesitamos también el ticker para el lookup en el backtest
predictions_df = predictions_df.join(df['ticker'])

output_path = 'predictions.csv'
predictions_df.to_csv(output_path)

end_time = time.time()
print(f"\n¡Predicciones guardadas en '{output_path}' en {end_time - start_time:.2f} segundos!")
print(f"Dimensiones del DataFrame de predicciones: {predictions_df.shape}")
print("\nPrimeras 5 filas de las predicciones:")
print(predictions_df.head())

