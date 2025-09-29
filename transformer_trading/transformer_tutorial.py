import matplotlib
matplotlib.use('Agg') # Usar el backend no interactivo para guardar gráficos
import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import backtrader as bt

# --- 1. Cargar y Preparar los Datos ---
df = pd.read_csv('yahoo/AAPL.csv', index_col='datetime', parse_dates=True)

# Añadir indicadores técnicos
df.ta.rsi(length=14, append=True)
df.ta.macd(fast=12, slow=26, signal=9, append=True)
df.ta.bbands(length=20, append=True)
df.ta.atr(length=14, append=True)
df.ta.stoch(length=14, append=True)
df.dropna(inplace=True)

features = [
    'close',
    'RSI_14',
    'MACD_12_26_9',
    'BBL_20_2.0',
    'BBM_20_2.0',
    'BBU_20_2.0',
    'ATRr_14',
    'STOCHk_14_3_3',
    'STOCHd_14_3_3'
]
# Corrección de nombres de columnas de bbands
for col in df.columns:
    if 'BBL_20_2.0' in col:
        features[3] = col
    if 'BBM_20_2.0' in col:
        features[4] = col
    if 'BBU_20_2.0' in col:
        features[5] = col

data = df[features]
n_features = len(features)

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

scaler_pred = MinMaxScaler(feature_range=(0, 1))
scaler_pred.fit_transform(data[['close']])

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60
X, y = create_sequences(scaled_data, SEQ_LENGTH)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# --- 2. Construir y Entrenar el Modelo Transformer ---
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    x = LayerNormalization(epsilon=1e-6)(inputs)
    x = MultiHeadAttention(key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)
    x = Dropout(dropout)(x)
    res = x + inputs

    x = LayerNormalization(epsilon=1e-6)(res)
    x = Dense(ff_dim, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(inputs.shape[-1])(x)
    return x + res

def build_transformer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0):
    inputs = Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = GlobalAveragePooling1D(data_format="channels_last")(x)
    for dim in mlp_units:
        x = Dense(dim, activation="relu")(x)
        x = Dropout(mlp_dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs)

model = build_transformer_model(
    (SEQ_LENGTH, n_features),
    head_size=128,
    num_heads=4,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[64],
    dropout=0.1,
    mlp_dropout=0.1
)

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))

# --- 3. Guardar el Modelo ---
model.save('transformer_trading/transformer_model.keras')

# --- 4. Evaluar el Modelo ---
predictions = model.predict(X_test)
predictions = scaler_pred.inverse_transform(predictions)
y_test_inv = scaler_pred.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(14, 5))
plot_index = data.index[train_size + SEQ_LENGTH:]
plt.plot(plot_index, y_test_inv, color='blue', label='Precio Real')
plt.plot(plot_index, predictions, color='red', label='Precio Predicho')
plt.title('Predicción de Precios con Transformer')
plt.xlabel('Fecha')
plt.ylabel('Precio')
plt.legend()
plt.savefig('transformer_trading/prediction_plot.png')

# --- 5. Estrategia en backtrader ---
class TransformerStrategy(bt.Strategy):
    def __init__(self):
        self.model = load_model('transformer_trading/transformer_model.keras')
        self.scaler = scaler
        self.scaler_pred = scaler_pred
        self.seq_length = SEQ_LENGTH

        self.data_close = self.datas[0].close
        self.rsi = bt.indicators.RSI(self.datas[0], period=14)
        self.macd = bt.indicators.MACD(self.datas[0], period_me1=12, period_me2=26, period_signal=9)
        self.bbands = bt.indicators.BollingerBands(self.datas[0], period=20)
        self.atr = bt.indicators.AverageTrueRange(self.datas[0], period=14)
        self.stoch = bt.indicators.Stochastic(self.datas[0], period=14)

    def next(self):
        if len(self) < self.seq_length + 26:
            return

        close_vals = self.data_close.get(size=self.seq_length)
        rsi_vals = self.rsi.get(size=self.seq_length)
        macd_vals = self.macd.macd.get(size=self.seq_length)
        bbl_vals = self.bbands.lines.bot.get(size=self.seq_length)
        bbm_vals = self.bbands.lines.mid.get(size=self.seq_length)
        bbu_vals = self.bbands.lines.top.get(size=self.seq_length)
        atr_vals = self.atr.get(size=self.seq_length)
        stochk_vals = self.stoch.lines.percK.get(size=self.seq_length)
        stochd_vals = self.stoch.lines.percD.get(size=self.seq_length)

        input_array = np.array([
            close_vals, rsi_vals, macd_vals, 
            bbl_vals, bbm_vals, bbu_vals, 
            atr_vals, 
            stochk_vals, stochd_vals
        ]).T

        scaled_input = self.scaler.transform(input_array)
        X_pred = np.array([scaled_input])

        prediction_scaled = self.model.predict(X_pred)
        prediction = self.scaler_pred.inverse_transform(prediction_scaled)[0][0]

        if prediction > self.data_close[0]:
            if not self.position:
                self.buy()
        elif prediction < self.data_close[0]:
            if self.position:
                self.sell()

cerebro = bt.Cerebro()
data_feed = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data_feed)
cerebro.addstrategy(TransformerStrategy)
cerebro.broker.setcash(100000.0)
cerebro.broker.setcommission(commission=0.001)

print(f'Capital Inicial: {cerebro.broker.getvalue()}')
cerebro.run()
print(f'Capital Final: {cerebro.broker.getvalue()}')

# Solución para evitar que el gráfico intente mostrarse en un entorno no interactivo
_original_show = plt.show
plt.show = lambda: None

fig = cerebro.plot(style='candlestick')[0][0]
fig.savefig('transformer_trading/backtest_plot.png')

# Restaurar la función original
plt.show = _original_show
