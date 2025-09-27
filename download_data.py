from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import datetime

import backtrader as bt

def run(args):
    cerebro = bt.Cerebro()

    # Create an instance of the IBStore
    store = bt.stores.IBStore(host=args.host, port=args.port, clientId=args.clientId)

    # Define the data feed
    data = store.getdata(
        dataname=args.dataname,
        timeframe=bt.TimeFrame.TFrame(args.timeframe),
        fromdate=datetime.datetime.strptime(args.fromdate, '%Y-%m-%d'),
        todate=datetime.datetime.strptime(args.todate, '%Y-%m-%d'),
        compression=args.compression
    )

    # Add the data to Cerebro
    cerebro.adddata(data)

    # Run Cerebro
    cerebro.run()

def parse_args():
    parser = argparse.ArgumentParser(
        description='Download Historical Data from IBKR')

    parser.add_argument('--dataname', '-d',
                        required=True,
                        help='Symbol to download')

    parser.add_argument('--timeframe', '-t',
                        default='days', choices=['days', 'weeks', 'months'],
                        help='Timeframe to download')

    parser.add_argument('--compression', '-c',
                        default=1, type=int,
                        help='Compression for the timeframe')

    parser.add_argument('--fromdate', '-f',
                        default='2020-01-01',
                        help='From date in YYYY-MM-DD format')

    parser.add_argument('--todate', '-o',
                        default='2021-01-01',
                        help='To date in YYYY-MM-DD format')

    parser.add_argument('--host',
                        default='127.0.0.1',
                        help='Host for the TWS/Gateway connection')

    parser.add_argument('--port',
                        default=7497, type=int,
                        help='Port for the TWS/Gateway connection')
    
    parser.add_argument('--clientId',
                        default=45, type=int,
                        help='Client ID for the TWS/Gateway connection')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
