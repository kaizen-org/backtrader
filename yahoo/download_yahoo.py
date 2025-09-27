from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import argparse
import yfinance as yf
import pandas as pd

def run(args):
    # Download data from Yahoo Finance
    ticker = yf.Ticker(args.dataname)
    df = ticker.history(start=args.fromdate, end=args.todate)

    # Prepare the dataframe for backtrader
    df.rename(columns={
        'Open': 'open',
        'High': 'high',
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }, inplace=True)

    # Add openinterest column
    df['openinterest'] = 0

    # The index should be named 'datetime'
    df.index.name = 'datetime'

    # Save to CSV
    outfile = f'{args.dataname}.csv'
    df.to_csv(outfile)
    print(f"Saved data to {outfile}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Download Historical Data from Yahoo Finance')

    parser.add_argument('--dataname', '-d',
                        required=True,
                        help='Ticker to download')

    parser.add_argument('--fromdate', '-f',
                        default='2020-01-01',
                        help='From date in YYYY-MM-DD format')

    parser.add_argument('--todate', '-o',
                        default='2021-01-01',
                        help='To date in YYYY-MM-DD format')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    run(args)
