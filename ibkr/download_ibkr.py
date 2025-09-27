import argparse
import time
import threading
import pandas as pd
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class IBKRDownloader(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.data = []
        self.data_received = threading.Event()

    def historicalData(self, reqId, bar):
        self.data.append([bar.date, bar.open, bar.high, bar.low, bar.close, bar.volume])

    def historicalDataEnd(self, reqId, start, end):
        super().historicalDataEnd(reqId, start, end)
        print(f"HistoricalDataEnd. ReqId: {reqId}, from {start} to {end}")
        self.data_received.set()

    def error(self, reqId, errorCode, errorString):
        print(f"Error: {reqId}, {errorCode}, {errorString}")

def run(args):
    app = IBKRDownloader()
    app.connect(args.host, args.port, clientId=0)

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    time.sleep(1)

    contract = Contract()
    contract.symbol = args.symbol
    contract.secType = args.sectype
    contract.exchange = args.exchange
    contract.currency = args.currency

    app.reqHistoricalData(1, contract, args.todate, args.duration, args.timeframe, "MIDPOINT", 1, 1, False, [])

    app.data_received.wait()

    app.disconnect()

    df = pd.DataFrame(app.data, columns=['datetime', 'open', 'high', 'low', 'close', 'volume'])
    df['datetime'] = pd.to_datetime(df['datetime'])
    df.set_index('datetime', inplace=True)

    outfile = f'{args.symbol}.{args.currency}.csv'
    df.to_csv(outfile)
    print(f"Saved data to {outfile}")

def parse_args():
    parser = argparse.ArgumentParser(description='Download Historical Data from IBKR')

    parser.add_argument('--host', default='127.0.0.1', help='Host of TWS/Gateway')
    parser.add_argument('--port', type=int, default=7497, help='Port of TWS/Gateway')
    parser.add_argument('--symbol', required=True, help='Symbol of the instrument')
    parser.add_argument('--sectype', required=True, help='Security type (e.g., STK, CASH, FUT)')
    parser.add_argument('--exchange', required=True, help='Exchange (e.g., SMART, IDEALPRO)')
    parser.add_argument('--currency', required=True, help='Currency (e.g., USD, EUR)')
    parser.add_argument('--todate', default='', help='End date in YYYYMMDD HH:MM:SS format')
    parser.add_argument('--duration', default='1 Y', help='Duration (e.g., 1 D, 1 W, 1 M, 1 Y)')
    parser.add_argument('--timeframe', default='1 day', help='Time frame (e.g., 1 min, 5 mins, 1 day)')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
