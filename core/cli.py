import argparse
import sys
import logging
import asyncio

def setup_parser():
    parser = argparse.ArgumentParser(description="FactorMiner V4 Global CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Miner Subcommand
    miner_parser = subparsers.add_parser("mine", help="Run the FactorMiner engine")
    miner_parser.add_argument("--miner", type=str, required=True, help="Name of the miner class/paradigm (e.g., GP, LLM)")
    miner_parser.add_argument("--config", type=str, required=True, help="Path to the JSON configuration file")
    miner_parser.add_argument("--iterations", type=int, default=None, help="Optional: Number of iterations (overrides config)")
    miner_parser.add_argument("--user-dir", type=str, default="user_workspace", help="Path to the user workspace directory")
    
    # Download Subcommand
    dl_parser = subparsers.add_parser("download", help="Download historical market data")
    dl_parser.add_argument("--exchange", type=str, default="binance", help="Exchange name (e.g., binance)")
    dl_parser.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols (e.g., BTC/USDT,ETH/USDT)")
    dl_parser.add_argument("--timeframes", type=str, default="1d", help="Comma-separated timeframes (e.g., 1m,1h,1d)")
    dl_parser.add_argument("--type", type=str, default="spot", choices=["spot", "futures"], help="Market type (spot or futures)")
    dl_parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    dl_parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    
    return parser

def main():
    parser = setup_parser()
    args = parser.parse_args()
    
    if args.command == "mine":
        from core.commands import mine as miner_main
        miner_main.run_miner(args)
    elif args.command == "download":
        from core.commands import download as dl_main
        asyncio.run(dl_main.run_downloader(args))
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()
