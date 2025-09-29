#!/usr/bin/env python
# coding: utf-8

# # Trading Algorítmico con GRU y Backtrader
# 
# Este notebook sigue el tutorial `gru_tutorial.md`. Aquí implementaremos el flujo de trabajo completo: desde la preparación de los datos y el entrenamiento del modelo GRU hasta la creación y ejecución de una estrategia de trading en `backtrader`.

# ## 1. Cargar y Preparar los Datos

# In[ ]:


# Importar las bibliotecas necesarias
import matplotlib
matplotlib.use('Agg') # Usar el backend no interactivo para guardar gráficos
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
import matplotlib.pyplot as plt

# Cargar el conjunto de datos de precios de acciones
df = pd.read_csv('../yahoo/AAPL.csv', index_col='datetime', parse_dates=True)

# --- 1.1 Añadir un conjunto completo de Indicadores Técnicos ---
df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.bbands(length=20, append=True)
df.ta.atr(length=14, append=True)
df.ta.stoch(length=14, append=True)

# Eliminar filas con valores NaN generados por los indicadores
df.dropna(inplace=True)

# Seleccionar las características (features) para el modelo con los nombres corregidos
features = [
    'close',
    'RSI_14',
    'MACD_12_26_9',
    'BBL_20_2.0_2.0',
    'BBM_20_2.0_2.0',
    'BBU_20_2.0_2.0',
    'ATRr_14',
    'STOCHk_14_3_3',
    'STOCHd_14_3_3'
]
data = df[features]

# Guardar el número de características para usarlo en la forma de entrada del modelo
n_features = len(features)
print(f'Número de características: {n_features}')

# Escalar los datos al rango [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Crear un segundo escalador solo para el precio de cierre, para invertir la predicción
scaler_pred = MinMaxScaler(feature_range=(0, 1))
scaler_pred.fit_transform(data[['close']])

# Función para crear secuencias de datos para el modelo GRU
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0]) # Predecir solo el precio de cierre (columna 0)
    return np.array(X), np.array(y)

# Definir la longitud de la secuencia
SEQ_LENGTH = 60
X, y = create_sequences(scaled_data, SEQ_LENGTH)

# Dividir los datos en conjuntos de entrenamiento y prueba
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Imprimir las formas para verificar
print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')


# ## 2. Construir y Entrenar el Modelo GRU

# In[ ]:


# Construir el modelo GRU usando Keras Sequential API
model = Sequential([
    # Primera capa GRU con 50 unidades, con la forma de entrada correcta
    GRU(50, return_sequences=True, input_shape=(SEQ_LENGTH, n_features)),
    # Capa de Dropout para regularización
    Dropout(0.2),
    # Segunda capa GRU con 50 unidades
    GRU(50, return_sequences=False),
    Dropout(0.2),
    # Capa densa con 25 neuronas
    Dense(25),
    # Capa de salida con una neurona para predecir el precio de cierre
    Dense(1)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error')

# Mostrar un resumen de la arquitectura del modelo
model.summary()

# Entrenar el modelo
history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))


# ## 3. Guardar el Modelo Entrenado

# In[ ]:


# Guardar el modelo entrenado en el formato recomendado .keras
model.save('gru_model.keras')
print('Modelo guardado como gru_model.keras')


# ## 4. Evaluar el Modelo

# In[ ]:


# Realizar predicciones sobre el conjunto de prueba
predictions = model.predict(X_test)
# Invertir la escala de las predicciones para obtener los precios reales
predictions = scaler_pred.inverse_transform(predictions)
# Invertir la escala de los datos de prueba para comparación
y_test_inv = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

# Graficar los precios reales vs. los precios predichos
plt.figure(figsize=(14, 5))
# Asegurarse de que el índice para el eje X es correcto
plot_index = data.index[train_size + SEQ_LENGTH:]
plt.plot(plot_index, y_test_inv, color='blue', label='Precio Real')
plt.plot(plot_index, predictions, color='red', label='Precio Predicho')
plt.title('Predicción de Precios de AAPL con Indicadores')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.savefig('prediction_plot.png') # Guardar el gráfico en lugar de mostrarlo
print('Gráfico de predicción guardado como prediction_plot.png')


# ## 5. Crear la Estrategia en `backtrader`

# In[ ]:


# Importar las bibliotecas necesarias para backtesting
import backtrader as bt
from tensorflow.keras.models import load_model

# Definir la estrategia de trading basada en el modelo GRU
class GRUStrategy(bt.Strategy):
    def __init__(self):
        # Cargar el modelo GRU pre-entrenado desde el nuevo formato
        self.model = load_model('gru_model.keras')
        # Mantener los escaladores para transformar los datos
        self.scaler = scaler
        self.scaler_pred = scaler_pred
        self.seq_length = SEQ_LENGTH
        self.n_features = n_features

        # --- Inicializar todos los indicadores usados en el entrenamiento ---
        self.data_close = self.datas[0].close
        self.rsi = bt.indicators.RSI(self.datas[0], period=14)
        self.macd = bt.indicators.MACD(self.datas[0], period_me1=12, period_me2=26, period_signal=9)
        self.bbands = bt.indicators.BollingerBands(self.datas[0], period=20)
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=14)
        self.stoch = bt.indicators.Stochastic(self.datas[0], period=14)

    def next(self):
        # Dejar que los indicadores se calienten. El periodo más largo es 26 (MACD) + 60 de la secuencia.
        if len(self) < self.seq_length + 26:
            return

        # --- Recoger los últimos `seq_length` valores de cada característica ---
        close_vals = self.data_close.get(size=self.seq_length)
        rsi_vals = self.rsi.get(size=self.seq_length)
        macd_vals = self.macd.macd.get(size=self.seq_length)
        bbl_vals = self.bbands.lines.bot.get(size=self.seq_length)
        bbm_vals = self.bbands.lines.mid.get(size=self.seq_length)
        bbu_vals = self.bbands.lines.top.get(size=self.seq_length)
        atr_vals = self.atr.get(size=self.seq_length)
        stochk_vals = self.stoch.lines.percK.get(size=self.seq_length)
        stochd_vals = self.stoch.lines.percD.get(size=self.seq_length)

        # Crear la matriz de entrada para el modelo en el orden correcto
        input_array = np.array([
            close_vals, rsi_vals, macd_vals, 
            bbl_vals, bbm_vals, bbu_vals, 
            atr_vals, 
            stochk_vals, stochd_vals
        ]).T

        # Escalar los datos de entrada
        scaled_input = self.scaler.transform(input_array)

        # Preparar el lote para la predicción
        X_pred = np.array([scaled_input])

        # Predecir el precio del siguiente día
        prediction_scaled = self.model.predict(X_pred)
        prediction = self.scaler_pred.inverse_transform(prediction_scaled)[0][0]

        # Lógica de trading
        if prediction > self.data_close[0]:
            if not self.position:
                self.buy()
        elif prediction < self.data_close[0]:
            if self.position:
                self.sell()

# --- Ejecutar el Backtest ---
cerebro = bt.Cerebro()
data_feed = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data_feed)
cerebro.addstrategy(GRUStrategy)
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.001)

print(f'Capital Inicial: {cerebro.broker.getvalue()}')
cerebro.run()
print(f'Capital Final: {cerebro.broker.getvalue()}')

# Guardar el gráfico del backtest con las operaciones
fig = cerebro.plot(style='candlestick')[0][0]
fig.savefig('backtest_plot.png')
print('Gráfico del backtest guardado como backtest_plot.png')

# Guardar el gráfico del backtest sin indicadores
# plotind=False evita que se dibujen los indicadores en sub-gráficos
fig2 = cerebro.plot(style='candlestick', plotind=False)[0][0]
fig2.savefig('backtest_plot_simple.png')
print('Gráfico del backtest sin indicadores guardado como backtest_plot_simple.png')

