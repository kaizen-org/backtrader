
'''
Fase 2: Construcción y Entrenamiento del Modelo GRU

Este script carga los datos enriquecidos, los prepara en secuencias y entrena
un modelo GRU para predecir la estrategia óptima (Momentum vs. Reversión a la Media).
'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import time

print("Iniciando el script de entrenamiento del modelo GRU...")
start_time = time.time()

# --- 2.1. Carga y Preparación de Datos ---
print("\nFase 2.1: Cargando y preparando datos...")

df = pd.read_csv('enriched_data.csv', index_col='datetime', parse_dates=True)

# Seleccionamos las características (features) y el objetivo (target)
features = [
    'open', 'high', 'low', 'close', 'volume',
    'SMA_20', 'SMA_50', 'SMA_200',
    'RSI_14',
    'BBL_20_2.0', 'BBM_20_2.0', 'BBU_20_2.0',
    'MACD_12_26_9', 'MACDh_12_26_9', 'MACDs_12_26_9',
    'ATRr_14',
    'daily_return'
]

target = 'estrategia_optima'

# Creamos los dataframes de features y target
df_features = df[features]
df_target = df[target]

# --- 2.2. División y Normalización de Datos ---
print("\nFase 2.2: Dividiendo y normalizando datos...")

# División cronológica (80% entrenamiento, 20% validación)
split_ratio = 0.8
split_index = int(len(df_features) * split_ratio)

X_train_raw = df_features[:split_index]
X_test_raw = df_features[split_index:]
y_train_raw = df_target[:split_index]
y_test_raw = df_target[split_index:]

print(f"Datos de entrenamiento: {len(X_train_raw)} filas")
print(f"Datos de validación: {len(X_test_raw)} filas")

# Normalización
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Guardar el scaler para usarlo en el backtesting
joblib.dump(scaler, 'gru_scaler.joblib')
print("Scaler guardado en 'gru_scaler.joblib'")

# --- 2.3. Creación de Secuencias ---
print("\nFase 2.3: Creando secuencias para la GRU...")

SEQUENCE_LENGTH = 30 # Usaremos los últimos 30 días para predecir el siguiente

def create_sequences(features, target, seq_length):
    X_seq, y_seq = [], []
    for i in range(len(features) - seq_length):
        X_seq.append(features[i:i+seq_length])
        y_seq.append(target[i+seq_length])
    return np.array(X_seq), np.array(y_seq)

# Crear secuencias para entrenamiento
X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train_raw.values, SEQUENCE_LENGTH)

# Crear secuencias para validación
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test_raw.values, SEQUENCE_LENGTH)

print(f"Shape de secuencias de entrenamiento (X): {X_train_seq.shape}")
print(f"Shape de secuencias de validación (X): {X_test_seq.shape}")

# --- 2.4. Construcción y Entrenamiento del Modelo GRU ---
print("\nFase 2.4: Construyendo y entrenando el modelo GRU...")

model = Sequential([
    GRU(50, return_sequences=True, input_shape=(SEQUENCE_LENGTH, X_train_seq.shape[2])),
    Dropout(0.2),
    GRU(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1, activation='sigmoid') # Salida binaria
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Usaremos EarlyStopping para evitar sobreajuste y detener el entrenamiento si no mejora
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X_train_seq,
    y_train_seq,
    epochs=50, # Un número alto, EarlyStopping decidirá cuándo parar
    batch_size=32,
    validation_data=(X_test_seq, y_test_seq),
    callbacks=[early_stopping],
    verbose=1
)

# --- 2.5. Evaluación y Guardado ---
print("\nFase 2.5: Evaluando y guardando el modelo...")

loss, accuracy = model.evaluate(X_test_seq, y_test_seq, verbose=0)
print(f"Precisión (Accuracy) en el conjunto de validación: {accuracy:.4f}")

# Guardar el modelo entrenado
model.save('gru_strategy_selector.h5')
print("Modelo guardado en 'gru_strategy_selector.h5'")

end_time = time.time()
print(f"\n¡Fase 2 completada en {end_time - start_time:.2f} segundos!")


