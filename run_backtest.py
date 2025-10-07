
'''
Fase 3 (Revisado): Backtesting con Predicciones Pre-calculadas

Este script ejecuta el backtest final de una forma mucho más eficiente.
En lugar de cargar el pesado modelo de TF dentro de la estrategia, carga
las predicciones ya generadas y las consulta en cada paso.
'''

import backtrader as bt
import pandas as pd
import os
from datetime import datetime

# --- Carga de Predicciones ---
PREDICTIONS_PATH = 'predictions.csv'

print(f"Cargando predicciones desde {PREDICTIONS_PATH}...")
predictions_df = pd.read_csv(PREDICTIONS_PATH, index_col='datetime', parse_dates=True)

# Obtener el rango de fechas para el backtest (out-of-sample)
start_date = predictions_df.index.min()
end_date = predictions_df.index.max()

print(f"El backtest se ejecutará en el rango de fechas: {start_date.date()} a {end_date.date()}")

# --- Definición de la Estrategia en Backtrader ---

class GRUHybridStrategy(bt.Strategy):
    params = (
        ('ticker', ''),
        ('predictions', None), # Pasamos el DataFrame de predicciones como parámetro
    )

    def __init__(self):
        print(f"Inicializando estrategia para {self.p.ticker}...")
        self.data_close = self.datas[0].close

        # Indicadores para la lógica de trading
        self.sma50 = bt.indicators.SimpleMovingAverage(self.datas[0], period=50)
        self.sma200 = bt.indicators.SimpleMovingAverage(self.datas[0], period=200)
        self.bband = bt.indicators.BollingerBands(self.datas[0], period=20)

        self.order = None

        # Filtra las predicciones para el ticker de esta instancia de la estrategia
        self.ticker_predictions = self.p.predictions[self.p.predictions['ticker'] == self.p.ticker]

    def log(self, txt, dt=None):
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} - {txt}')

    def next(self):
        current_date = self.datas[0].datetime.date(0)

        # --- Búsqueda de la Predicción ---
        try:
            # Busca la estrategia pre-calculada para el día actual
            predicted_strategy = self.ticker_predictions.loc[current_date.strftime('%Y-%m-%d')]['predicted_strategy']
        except KeyError:
            # No hay predicción para este día, no hacemos nada
            return

        # --- Lógica de Trading ---
        if self.order:
            return

        # Comprobación de posición específica para este activo
        if not self.getposition(self.data).size:
            if predicted_strategy == 1: # Estrategia de Momentum
                if self.sma50 > self.sma200:
                    self.log(f'MODELO PREDICE MOMENTUM. COMPRA {self.p.ticker} a {self.data_close[0]:.2f}')
                    self.order = self.buy()
            else: # Estrategia de Reversión a la Media
                if self.data_close[0] < self.bband.lines.bot[0]:
                    self.log(f'MODELO PREDICE REVERSIÓN. COMPRA {self.p.ticker} a {self.data_close[0]:.2f}')
                    self.order = self.buy()
        # Si ya tenemos una posición en ESTE activo
        elif self.getposition(self.data).size:
            if predicted_strategy == 1: # Lógica de salida para Momentum
                if self.sma50 <= self.sma200:
                    self.log(f'SALIDA MOMENTUM. VENTA {self.p.ticker} a {self.data_close[0]:.2f}')
                    self.order = self.sell()
            else: # Lógica de salida para Reversión a la Media
                if self.data_close[0] > self.bband.lines.mid[0]:
                    self.log(f'SALIDA REVERSIÓN. VENTA {self.p.ticker} a {self.data_close[0]:.2f}')
                    self.order = self.sell()

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f'COMPRA EJECUTADA, Precio: {order.executed.price:.2f}, Coste: {order.executed.value:.2f}, Comisión: {order.executed.comm:.2f}')
            elif order.issell():
                self.log(f'VENTA EJECUTADA, Precio: {order.executed.price:.2f}, Beneficio: {order.executed.pnl:.2f}, Comisión: {order.executed.comm:.2f}')
            self.bar_executed = len(self)
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Orden Cancelada/Rechazada/Margen')
        self.order = None

# --- Configuración y Ejecución del Backtest ---
if __name__ == '__main__':
    print("\nIniciando backtest con predicciones pre-calculadas...")
    cerebro = bt.Cerebro()

    assets_to_backtest = ['A.USD.csv', 'ABM.USD.csv']
    for asset_file in assets_to_backtest:
        ticker_name = asset_file.split('.')[0]
        data = bt.feeds.GenericCSVData(
            dataname=os.path.join('ibkr', asset_file),
            dtformat=('%Y-%m-%d'),
            fromdate=start_date,
            todate=end_date,
            datetime=0, open=1, high=2, low=3, close=4, volume=5,
            openinterest=-1
        )
        cerebro.adddata(data, name=ticker_name)
        cerebro.addstrategy(GRUHybridStrategy, ticker=ticker_name, predictions=predictions_df)

    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.001)
    # Usar un tamaño fijo para depurar y obtener resultados realistas
    cerebro.addsizer(bt.sizers.FixedSize, stake=100) 

    print('Capital Inicial: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print('Capital Final: %.2f' % cerebro.broker.getvalue())

    print("Generando gráfico...")
    cerebro.plot(style='candlestick', barup='green', bardown='red', savefig=True, figfilename='backtest_plot.png')
