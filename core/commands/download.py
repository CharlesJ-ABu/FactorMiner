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
    if not all(symbols) or not all(timeframes):
        logging.error("Symbols and timeframes must contain non-empty entries.")
        raise SystemExit(1)
    
    downloader = SmartBatchDownloader()
    results = []
    
    for symbol in symbols:
        for timeframe in timeframes:
            logging.info(f"Downloading {symbol} {timeframe} ({args.type}) from {args.start} to {args.end} (mode: {args.mode})...")
            try:
                result = await asyncio.to_thread(
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
                success = isinstance(result, dict) and result.get("success") is True
                error = None if success else (
                    result.get("error", "Downloader returned no successful result")
                    if isinstance(result, dict) else "Downloader returned no successful result"
                )
            except Exception as exc:
                success, error = False, str(exc)
            results.append({"symbol": symbol, "timeframe": timeframe, "trade_type": args.type,
                            "success": success, "error": error})
            if not success:
                logging.error("Download failed for %s %s (%s): %s", symbol, timeframe, args.type, error)
            
    failures = [result for result in results if not result["success"]]
    if failures:
        logging.error("Batch download failed: %d/%d targets failed.", len(failures), len(results))
        raise SystemExit(1)
    logging.info("Batch download completed: all %d targets succeeded.", len(results))
    return results

if __name__ == "__main__":
    args = parse_args()
    asyncio.run(run_downloader(args))
