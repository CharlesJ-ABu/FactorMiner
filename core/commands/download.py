import argparse
import logging
import asyncio
from core.data_feed.batch_downloader import SmartBatchDownloader

def parse_args():
    parser = argparse.ArgumentParser(description="FactorMiner V4 CLI - Data Downloader")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name (e.g., binance)")
    parser.add_argument("--symbols", type=str, required=True, help="Comma-separated symbols (e.g., BTC/USDT,ETH/USDT)")
    parser.add_argument("--timeframes", type=str, default="1d", help="Comma-separated timeframes (e.g., 1m,1h,1d)")
    parser.add_argument("--type", type=str, default="spot", choices=["spot", "futures"], help="Market type (spot or futures)")
    parser.add_argument("--start", type=str, required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "incremental", "overwrite"], help="Download mode (full, incremental, or overwrite)")
    return parser.parse_args()

async def run_downloader(args):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    
    downloader = SmartBatchDownloader()
    
    for symbol in symbols:
        for timeframe in timeframes:
            logging.info(f"Downloading {symbol} {timeframe} ({args.type}) from {args.start} to {args.end} (mode: {args.mode})...")
            await asyncio.to_thread(
                downloader.download_ohlcv_batch,
                exchange_id=args.exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=args.start,
                end_date=args.end,
                trade_type=args.type,
                progress_callback=None,
                download_mode=args.mode
            )
            
    logging.info("Batch download completed.")

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_downloader(args))
