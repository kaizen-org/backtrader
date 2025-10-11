import argparse
import time
import threading
import pandas as pd
import sys
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBKRDownloader(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.data_received = threading.Event()
        self.contract_details_received = threading.Event()
        self.contracts = []

    def historicalData(self, reqId, bar):
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])

    def historicalDataEnd(self, reqId, start, end):
        super().historicalDataEnd(reqId, start, end)
        print(f"HistoricalDataEnd. ReqId: {reqId}, from {start} to {end}")
        self.data_received.set()

    def contractDetails(self, reqId, contractDetails):
        self.contracts.append(contractDetails.contract)

    def contractDetailsEnd(self, reqId):
        super().contractDetailsEnd(reqId)
        self.contract_details_received.set()

    def error(self, reqId, errorCode, errorString):
        if errorCode != 200:
            print(f"Error: {reqId}, {errorCode}, {errorString}")

def run(args):
    app = IBKRDownloader()
    app.connect(args.host, args.port, clientId=args.clientid)

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    time.sleep(1)

    search_contract = Contract()
    search_contract.symbol = args.symbol
    search_contract.secType = args.sectype
    search_contract.currency = args.currency
    search_contract.exchange = args.exchange

    print(f"Buscando contrato para el símbolo: {args.symbol}")
    app.reqContractDetails(1, search_contract)
    
    if not app.contract_details_received.wait(10):
        print(f"Timeout esperando detalles del contrato para {args.symbol}")
        app.disconnect()
        sys.exit(1)

    if not app.contracts:
        print(f"LOG: No se encontró ningún contrato para el símbolo: {args.symbol}. Omitiendo.")
        app.disconnect()
        sys.exit(1)

    final_contract = app.contracts[0]
    if len(app.contracts) > 1:
        print(f"LOG: Se encontraron múltiples contratos para {args.symbol}. Seleccionando el más probable.")
        preferred_exchanges = ["NYSE", "NASDAQ", "ARCA", "AMEX"]
        for c in app.contracts:
            if c.primaryExchange in preferred_exchanges:
                final_contract = c
                break
        print(f"LOG: Contrato seleccionado: {final_contract.symbol} en {final_contract.primaryExchange}")

    print(f"Solicitando datos históricos para {final_contract.symbol}...")
    app.reqHistoricalData(2, final_contract, args.todate, args.duration, args.timeframe, "MIDPOINT", 1, 1, False, [])

    if not app.data_received.wait(30):
        print(f"Timeout esperando datos históricos para {final_contract.symbol}")
        app.disconnect()
        sys.exit(1)

    app.disconnect()

    if not app.data:
        print(f"No se recibieron datos para {final_contract.symbol}.")
        sys.exit(1)

    df = pd.DataFrame(app.data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    outfile = f'{args.symbol.replace(" ", ".")}.{args.currency}.csv'
    df.to_csv(outfile)
    print(f"Datos guardados en {outfile}")
    sys.exit(0)

def parse_args():
    parser = argparse.ArgumentParser(description='Download Historical Data from IBKR')

    parser.add_argument('--host', default='127.0.0.1', help='Host of TWS/Gateway')
    parser.add_argument('--port', type=int, default=7496, help='Port of TWS/Gateway')
    parser.add_argument('--symbol', required=True, help='Symbol of the instrument')
    parser.add_argument('--sectype', required=True, help='Security type (e.g., STK, CASH, FUT)')
    parser.add_argument('--exchange', required=True, help='Exchange (e.g., SMART, IDEALPRO)')
    parser.add_argument('--currency', required=True, help='Currency (e.g., USD, EUR)')
    parser.add_argument('--todate', default='', help='End date in YYYYMMDD HH:MM:SS format')
    parser.add_argument('--duration', default='1 Y', help='Duration (e.g., 1 D, 1 W, 1 M, 1 Y)')
    parser.add_argument('--timeframe', default='1 day', help='Time frame (e.g., 1 min, 5 mins, 1 day)')
    parser.add_argument('--clientid', type=int, default=101, help='Unique client ID for the connection')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f"Conectando a {args.host}:{args.port} con ClientID {args.clientid}")
    try:
        run(args)
    except Exception as e:
        print(f"Ocurrió una excepción no controlada: {e}")
        sys.exit(1)
