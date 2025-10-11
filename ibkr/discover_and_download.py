import argparse
import time
import threading
import subprocess
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract

class ContractDiscoverer(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.contracts = []
        self.discovery_done = threading.Event()

    def contractDetails(self, reqId, contractDetails):
        self.contracts.append(contractDetails.contract)

    def contractDetailsEnd(self, reqId):
        super().contractDetailsEnd(reqId)
        print("Contract discovery finished.")
        self.discovery_done.set()

    def error(self, reqId, errorCode, errorString):
        # Ignore informational messages about connectivity
        if errorCode in [2104, 2106, 2158]:
            print(f"Info: {errorCode} - {errorString}")
        else:
            print(f"Error: {reqId}, {errorCode}, {errorString}")

def run_discovery(args):
    app = ContractDiscoverer()
    app.connect(args.host, args.port, clientId=args.clientid)

    api_thread = threading.Thread(target=app.run, daemon=True)
    api_thread.start()
    # Wait for connection to be established
    time.sleep(2) 

    if not app.isConnected():
        print("Failed to connect to TWS/Gateway.")
        return []

    search_contract = Contract()
    search_contract.secType = args.sectype
    search_contract.exchange = args.exchange
    search_contract.currency = args.currency
    if args.symbol:
        search_contract.symbol = args.symbol

    print(f"Searching for contract(s) with criteria: SecType={args.sectype}, Exchange={args.exchange}, Currency={args.currency}, Symbol={args.symbol or 'Any'}")
    app.reqContractDetails(1, search_contract)

    # Wait for discovery to complete, with a timeout
    app.discovery_done.wait(timeout=15) 
    app.disconnect()
    
    return app.contracts

def download_contract_data(contract, args):
    print(f"--- Downloading data for {contract.symbol} ---")
    # Use a different client ID for the download subprocess
    download_client_id = args.clientid + 1
    command = [
        'python',
        'ibkr/download_ibkr.py',
        '--symbol', contract.symbol,
        '--sectype', contract.secType,
        '--exchange', contract.exchange,
        '--currency', contract.currency,
        '--host', str(args.host),
        '--port', str(args.port),
        '--clientid', str(download_client_id)
    ]
    try:
        result = subprocess.run(command, check=True, text=True, capture_output=True, timeout=60)
        print(f"Successfully downloaded {contract.symbol}")
        print(f"Output:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to download {contract.symbol}.")
        print(f"Stderr:\n{e.stderr}")
    except subprocess.TimeoutExpired:
        print(f"Timeout expired while trying to download {contract.symbol}.")

def parse_args():
    parser = argparse.ArgumentParser(description='Discover and Download Historical Data from IBKR')

    # Connection args
    parser.add_argument('--host', default='127.0.0.1', help='Host of TWS/Gateway')
    parser.add_argument('--port', type=int, default=7497, help='Port of TWS/Gateway')
    parser.add_argument('--clientid', type=int, default=102, help='Unique client ID for the connection')

    # Contract search args
    parser.add_argument('--sectype', required=True, help='Security type to search for (e.g., STK, CASH, FUT)')
    parser.add_argument('--exchange', default='SMART', help='Exchange to search on (e.g., SMART, IDEALPRO)')
    parser.add_argument('--currency', default='USD', help='Currency to search for (e.g., USD, EUR)')
    parser.add_argument('--symbol', default='', help='Optional: Specific symbol to search for')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    all_discovered_contracts = []

    # If a specific symbol is provided by the user, just search for that.
    if args.symbol:
        print(f"--- Searching for specific symbol: {args.symbol} ---")
        all_discovered_contracts = run_discovery(args)
    # If no symbol is provided, loop through the alphabet.
    else:
        print("--- Starting discovery for all symbols (A-Z) ---")
        for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            print(f"--- Searching for symbols starting with: {letter} ---")
            
            # Create a temporary args object for the discovery run
            search_args = argparse.Namespace(**vars(args))
            search_args.symbol = letter
            
            contracts_for_letter = run_discovery(search_args)
            all_discovered_contracts.extend(contracts_for_letter)
            
            # It's good practice to wait between broad requests to avoid pacing violations.
            print(f"Found {len(contracts_for_letter)} contracts starting with {letter}. Waiting 5 seconds...")
            time.sleep(5)

    if not all_discovered_contracts:
        print("\nNo contracts found for the given criteria. This could be due to a too-specific search, or not having market data subscriptions for the instrument.")
    else:
        print(f"\nTotal contracts found: {len(all_discovered_contracts)}. Starting download process...\n")
        for i, contract in enumerate(all_discovered_contracts):
            print(f"Downloading {i+1}/{len(all_discovered_contracts)}: {contract.symbol}")
            download_contract_data(contract, args)
        print("\nAll downloads finished.")
