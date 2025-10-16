import pandas as pd
import numpy as np
import os
import sys # Importar sys para la detección del entorno
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import joblib
import matplotlib.pyplot as plt

# --- CONSTANTES ---
SEQUENCE_LENGTH = 30
TRAIN_TEST_SPLIT_RATIO = 0.85 # 85% para entrenamiento, 15% para validación

def create_sequences(data, sequence_length):
    '''Crea secuencias a partir de los datos de un único ticker.'''
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:(i + sequence_length), :-1]) # Todas las columnas menos la última (target)
        y.append(data[i + sequence_length - 1, -1]) # El target del último día de la secuencia
    return np.array(X), np.array(y)

if __name__ == '__main__':
    print("Iniciando Fase 2: Entrenamiento del Modelo GRU de Correlación.")
    start_time = time.time()

    # --- Detección de Entorno y Configuración de Rutas ---
    IN_COLAB = 'google.colab' in sys.modules

    if IN_COLAB:
        print("Entorno de Google Colab detectado. Montando Google Drive...")
        from google.colab import drive
        drive.mount('/content/drive')
        
        # Asume que los datos y el output estarán en esta carpeta dentro de Drive
        BASE_DIR = '/content/drive/MyDrive/backtrader_colab'
        OUTPUT_DIR = os.path.join(BASE_DIR, 'correlation_output')
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        DATA_PATH = os.path.join(BASE_DIR, 'all_enriched_data.csv')
        MODEL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'correlation_model.h5')
        SCALER_OUTPUT_PATH = os.path.join(OUTPUT_DIR, 'correlation_scaler.joblib')
        TRAINING_LOSS_PLOT_PATH = os.path.join(OUTPUT_DIR, 'training_loss.png')
        
        print(f"Ruta de datos: {DATA_PATH}")
        print(f"Directorio de salida: {OUTPUT_DIR}")
    else:
        print("Entorno local detectado.")
        # Cambiar al directorio del script para que las rutas relativas funcionen
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        
        DATA_PATH = 'all_enriched_data.csv'
        MODEL_OUTPUT_PATH = 'correlation_model.h5'
        SCALER_OUTPUT_PATH = 'correlation_scaler.joblib'
        TRAINING_LOSS_PLOT_PATH = 'training_loss.png'

    # --- 1. Carga de Datos ---
    if not os.path.exists(DATA_PATH):
        print(f"Error: No se encuentra el archivo '{DATA_PATH}'.")
        if IN_COLAB:
            print("Asegúrate de que el archivo está en la ruta correcta de Google Drive.")
        else:
            print("Ejecuta 'data_prep_correlation.py' primero.")
        exit()
    
    print(f"Cargando datos desde '{DATA_PATH}'...")
    df = pd.read_csv(DATA_PATH, index_col='datetime', parse_dates=True)

    # --- 2. Preparación de Features y Target ---
    # El target es la última columna, las features son todas las demás excepto 'ticker'
    feature_columns = [col for col in df.columns if col not in ['ticker', 'future_return_5d']]
    target_column = 'future_return_5d'
    
    # Reordenar para que el target esté al final
    df_processed = df[feature_columns + [target_column]]

    print(f"Usando {len(feature_columns)} features.")

    # --- 3. Creación de Secuencias por Ticker ---
    print("Creando secuencias para cada ticker...")
    all_X, all_y = [], []
    grouped = df.groupby('ticker')
    print("Loop iniciado")
    for ticker_name, ticker_data in grouped:
        print(f"Procesando ticker: {ticker_name}")
        # Convertir el dataframe del ticker a numpy array
        ticker_values = ticker_data[feature_columns + [target_column]].values
        print(f"  - Shape de los datos del ticker: {ticker_values.shape}")
        
        if len(ticker_values) > SEQUENCE_LENGTH:
            print(f"  - Longitud de datos ({len(ticker_values)}) > SEQUENCE_LENGTH ({SEQUENCE_LENGTH}). Creando secuencias.")
            X_ticker, y_ticker = create_sequences(ticker_values, SEQUENCE_LENGTH)
            all_X.append(X_ticker)
            all_y.append(y_ticker)
            print(f"  - Secuencias creadas para {ticker_name}. Shape X: {X_ticker.shape}, Shape y: {y_ticker.shape}")
        else:
            print(f"  - Longitud de datos ({len(ticker_values)}) <= SEQUENCE_LENGTH ({SEQUENCE_LENGTH}). Saltando ticker.")

    print("Loop finalizado")
    
    if not all_X:
        print("Error: No se pudieron crear secuencias. Verifica los datos de entrada y el valor de SEQUENCE_LENGTH.")
        exit()

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"Secuencias creadas. Shape de X: {X.shape}, Shape de y: {y.shape}")

    # --- 4. División y Escalado de Datos ---
    # La división se hace por tiempo, no aleatoria, para evitar 'data leakage'
    split_index = int(X.shape[0] * TRAIN_TEST_SPLIT_RATIO)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    print(f"División de datos: {len(X_train)} para entrenamiento, {len(X_val)} para validación.")

    # Escalar solo las features (X)
    # Aplanar para el scaler: (samples * timesteps, features)
    nsamples, nx, ny = X_train.shape
    X_train_reshaped = X_train.reshape((nsamples * nx, ny))
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(X_train_reshaped)
    
    # Guardar el scaler
    joblib.dump(scaler, SCALER_OUTPUT_PATH)
    print(f"Scaler guardado en '{SCALER_OUTPUT_PATH}'.")

    # Transformar los datos de entrenamiento y validación
    X_train_scaled_reshaped = scaler.transform(X_train_reshaped)
    X_train_scaled = X_train_scaled_reshaped.reshape(X_train.shape)

    nsamples_val, nx_val, ny_val = X_val.shape
    X_val_reshaped = X_val.reshape((nsamples_val * nx_val, ny_val))
    X_val_scaled_reshaped = scaler.transform(X_val_reshaped)
    X_val_scaled = X_val_scaled_reshaped.reshape(X_val.shape)

    # --- 5. Construcción y Entrenamiento del Modelo GRU ---
    print("Construyendo el modelo GRU...")
    model = Sequential([
        GRU(units=100, return_sequences=True, input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2]), activation='tanh', recurrent_activation='sigmoid'),
        Dropout(0.2),
        GRU(units=50, return_sequences=False, activation='tanh', recurrent_activation='sigmoid'),
        Dropout(0.2),
        Dense(units=25, activation='relu'),
        Dense(units=1, activation='linear') # Salida lineal para regresión
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    # Callbacks para el entrenamiento
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(MODEL_OUTPUT_PATH, monitor='val_loss', save_best_only=True)

    print("\nIniciando entrenamiento...")
    history = model.fit(
        X_train_scaled, y_train,
        epochs=100,
        batch_size=64,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )

    # --- 6. Visualización del Entrenamiento ---
    print("Generando gráfica de la pérdida de entrenamiento...")
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Pérdida de Entrenamiento')
    plt.plot(history.history['val_loss'], label='Pérdida de Validación')
    plt.title('Curvas de Pérdida del Modelo')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida (MSE)')
    plt.legend()
    plt.grid(True)
    plt.savefig(TRAINING_LOSS_PLOT_PATH)
    print(f"Gráfica guardada como '{TRAINING_LOSS_PLOT_PATH}'.")

    end_time = time.time()
    print(f"\n¡Fase 2 completada en {end_time - start_time:.2f} segundos!")
    print(f"Modelo guardado en '{MODEL_OUTPUT_PATH}'.")