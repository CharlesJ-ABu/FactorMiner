from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.ws_manager import manager
import asyncio
import random
import time
import os
import datetime
import traceback

app = FastAPI(title="FactorMiner V4 API")

class TaskManager:
    tasks = {}

class LaunchRequest(BaseModel):
    miner: str
    config: str

class DownloadRequest(BaseModel):
    exchange: str
    symbols: list[str]
    timeframes: list[str]
    start_date: str
    end_date: str
    trade_types: list[str] = ["futures"]
    download_mode: str = "merge"

# Configure CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "engine": "FactorMinerDirector"}

@app.get("/api/miners")
async def get_miners():
    from core.utils.dynamic_loader import load_user_modules
    from core.miner.registry import MinerRegistry
    
    # Load custom modules
    load_user_modules("user_workspace")
    
    # Only return registered custom miners
    custom_miners = list(MinerRegistry._registry.keys())
    
    return {"miners": custom_miners}

@app.get("/api/configs")
async def get_configs():
    import json
    config_dir = os.path.join("user_workspace", "configs")
    if not os.path.exists(config_dir):
        return {"configs": {}}
    
    configs_data = {}
    for f in os.listdir(config_dir):
        if f.endswith(".json"):
            try:
                with open(os.path.join(config_dir, f), 'r') as file:
                    configs_data[f] = json.load(file)
            except Exception as e:
                configs_data[f] = {"error": str(e)}
                
    return {"configs": configs_data}

@app.get("/api/exchange_meta")
async def get_exchange_meta(exchange: str, trade_type: str = "futures"):
    # Try to fetch from CCXT, fallback if it fails (e.g., 451 error)
    import traceback
    
    meta = {
        "symbols": [
            "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT", 
            "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT", "DOT/USDT", "LTC/USDT",
            "BCH/USDT", "TRX/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT", "TON/USDT",
            "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT", "SEI/USDT",
            "TIA/USDT", "INJ/USDT", "FIL/USDT", "LDO/USDT", "RNDR/USDT", "STX/USDT",
            "ORDI/USDT", "PEPE/USDT", "SHIB/USDT", "WLD/USDT", "GALA/USDT", "FTM/USDT"
        ],
        "timeframes": ["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
        "trade_types": ["spot", "futures"],
        "min_date": "2017-01-01"
    }
    
    try:
        from core.data_feed.data_downloader import DataDownloader
        downloader = DataDownloader()
        ex_instance = downloader.get_exchange_instance(exchange_id=exchange, trade_type=trade_type)
        
        # We can optionally call ex_instance.load_markets() if it works in the region
        try:
            ex_instance.load_markets()
            if ex_instance.markets:
                # filter to some USDT pairs
                usdt_markets = [s for s in ex_instance.markets.keys() if s.endswith('/USDT') or s.endswith('USDT')]
                if usdt_markets:
                    try:
                        tickers = ex_instance.fetch_tickers()
                        # Sort by quoteVolume (24h volume) descending
                        sorted_markets = sorted(
                            usdt_markets, 
                            key=lambda s: float(tickers.get(s, {}).get('quoteVolume', 0) or 0), 
                            reverse=True
                        )
                        meta["symbols"] = sorted_markets[:200]  # limit to 200 to keep UI fast
                    except Exception:
                        meta["symbols"] = sorted(list(set(usdt_markets)))[:200]
            if ex_instance.timeframes:
                meta["timeframes"] = list(ex_instance.timeframes.keys())
        except Exception:
            pass # fallback to defaults if 451
            
    except Exception as e:
        print(f"Error fetching CCXT meta: {e}")
        
    return meta

@app.get("/api/tasks")
async def get_tasks():
    # Return all tasks sorted by start_time descending
    sorted_tasks = sorted(TaskManager.tasks.values(), key=lambda x: x["start_time"], reverse=True)
    return {"tasks": sorted_tasks}

@app.get("/api/stats")
async def get_stats():
    # Return global metrics
    tasks = list(TaskManager.tasks.values())
    total_tasks = len(tasks)
    
    # Check total factors from storage
    from core.storage.factor_storage import get_global_storage
    storage = get_global_storage()
    try:
        total_factors = len(storage.get_all_logic_hashes())
    except:
        total_factors = 0
        
    completed_tasks = sum(1 for t in tasks if t["status"] == "completed")
    success_rate = f"{(completed_tasks / total_tasks * 100):.1f}%" if total_tasks > 0 else "N/A"
    
    # Recent activity
    recent_tasks = sorted(tasks, key=lambda x: x["start_time"], reverse=True)[:5]
    
    return {
        "engine_online": True,
        "total_tasks": total_tasks,
        "total_factors": total_factors,
        "success_rate": success_rate,
        "recent_activity": recent_tasks
    }

@app.post("/api/launch")
async def launch_mining(req: LaunchRequest, background_tasks: BackgroundTasks):
    task_id = f"T-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}-{random.randint(100, 999)}"
    
    task_data = {
        "id": task_id,
        "status": "running",
        "miner": req.miner,
        "config": req.config,
        "progress": 0,
        "start_time": datetime.datetime.now().isoformat(),
        "error_msg": None,
        "hash": "---",
        "duration": "0s"
    }
    TaskManager.tasks[task_id] = task_data
    
    # Broadcast new task immediately
    await manager.broadcast({"type": "task_update", "task": task_data})
    
    background_tasks.add_task(run_mining_task_background, task_id, req.miner, req.config)
    return {"task_id": task_id}

async def run_mining_task_background(task_id: str, miner_name: str, config_name: str):
    import json
    import os
    import time
    import logging
    from core.miner.director import FactorMinerDirector
    
    start_time = time.time()
    task = TaskManager.tasks[task_id]
    main_loop = asyncio.get_running_loop()
    
    # Create a custom logging handler to broadcast real logs to the UI
    class WebsocketLogHandler(logging.Handler):
        def emit(self, record):
            log_entry = self.format(record)
            asyncio.run_coroutine_threadsafe(
                manager.broadcast({
                    "task_id": task_id,
                    "type": "log",
                    "text": log_entry
                }),
                main_loop
            )
            
    ws_handler = WebsocketLogHandler()
    ws_handler.setFormatter(logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
    ws_handler.setLevel(logging.INFO)
    
    # Attach handler to root logger temporarily
    root_logger = logging.getLogger()
    root_logger.addHandler(ws_handler)
    
    def progress_callback(epoch, max_epoch, best_factor):
        progress = int((epoch / max_epoch) * 100) if max_epoch > 0 else 0
        task["progress"] = progress
        
        elapsed = time.time() - start_time
        task["duration"] = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        
        if best_factor:
            task["hash"] = getattr(best_factor, 'logic_hash', 'N/A')
            
            # Optionally emit scatter data here too based on best_factor metrics
            ic = best_factor.metrics.get("IC", 0) if hasattr(best_factor, "metrics") else 0
            asyncio.run_coroutine_threadsafe(
                manager.broadcast({
                    "task_id": task_id,
                    "type": "scatter",
                    "epoch": epoch,
                    "ic": ic,
                    "complexity": 5  # mock complexity or read from factor
                }),
                main_loop
            )
            
        # Safely broadcast from synchronous thread back to main event loop
        asyncio.run_coroutine_threadsafe(
            manager.broadcast({"type": "task_update", "task": task}),
            main_loop
        )

    try:
        config_path = os.path.join("user_workspace", "configs", config_name)
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        config["paradigm"] = miner_name
        
        from core.data_feed.real_client import RealDataClient
        data_client = RealDataClient(config)
        
        director = FactorMinerDirector(config, data_client)
        max_iter = config.get("max_iterations", 10)
        
        # We must run director.run in a separate thread to avoid blocking FastAPI
        best_factors = await asyncio.to_thread(
            director.run, max_iterations=max_iter, progress_callback=progress_callback
        )
        
        task["status"] = "completed"
        task["progress"] = 100
        if best_factors:
            task["hash"] = getattr(best_factors[0], 'logic_hash', 'N/A')
            
    except Exception as e:
        task["status"] = "failed"
        task["error_msg"] = str(e) + "\n" + traceback.format_exc()
        
    finally:
        root_logger.removeHandler(ws_handler)
        elapsed = time.time() - start_time
        task["duration"] = f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        try:
            # Need a hack to broadcast from this async context
            await manager.broadcast({"type": "task_update", "task": task})
        except Exception:
            pass

class BatchCoverageRequest(BaseModel):
    exchange: str
    symbols: list[str]
    timeframes: list[str]
    trade_types: list[str]

@app.post("/api/batch_data_coverage")
async def get_batch_data_coverage(req: BatchCoverageRequest):
    results = []
    for symbol in req.symbols:
        for timeframe in req.timeframes:
            for trade_type in req.trade_types:
                coverage = await get_data_coverage(req.exchange, symbol, timeframe, trade_type)
                results.append({
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "trade_type": trade_type,
                    "coverage": coverage
                })
    return {"results": results}

@app.get("/api/data_coverage")
async def get_data_coverage(exchange: str, symbol: str, timeframe: str, trade_type: str = "futures"):
    import pandas as pd
    from pathlib import Path
    
    # 构建文件名
    safe_symbol = symbol.replace('/', '_').replace(':', '_')
    dir_path = Path(f"data/{exchange}/{trade_type}")
    if not dir_path.exists():
        dir_path = Path(f"data/{exchange}")
        
    if not dir_path.exists():
        return {"exists": False, "message": "Directory not found"}
        
    matched_files = []
    
    if trade_type in ['futures', 'spot']:
        # 严格按照 batch_downloader.py 的本地数据存储命名进行匹配
        exact_filename = f"{safe_symbol}-{timeframe}-{trade_type}.feather"
        target_file = dir_path / exact_filename
        
        if target_file.exists():
            matched_files = [target_file]
        else:
            # Fallback
            search_pattern = f"{safe_symbol}*-{timeframe}-{trade_type}.feather"
            alt_pattern = f"{safe_symbol}_{timeframe}_*.feather"
            matched_files = list(dir_path.glob(search_pattern)) + list(dir_path.glob(alt_pattern))
    else:
        search_pattern = f"{safe_symbol}_{timeframe}_*.feather"
        matched_files = list(dir_path.glob(search_pattern))
    
    if not matched_files:
        return {"exists": False, "message": "No data found"}
        
    # 取第一个匹配的文件读取 coverage
    target_file = matched_files[0]
    try:
        df = pd.read_feather(target_file)
        if len(df) == 0:
            return {"exists": False, "message": "File empty"}
            
        # 根据时间戳转换
        if 'date' in df.columns:
            if df['date'].dtype in ['int64', 'int32', 'float64']:
                sample = df['date'].iloc[0]
                if sample > 1e12:
                    df['date'] = pd.to_datetime(df['date'], unit='ms')
                else:
                    df['date'] = pd.to_datetime(df['date'], unit='s')
            
            start = df['date'].min().strftime('%Y-%m-%d %H:%M')
            end = df['date'].max().strftime('%Y-%m-%d %H:%M')
            return {
                "exists": True,
                "start_date": start,
                "end_date": end,
                "total_records": len(df),
                "filepath": str(target_file)
            }
    except Exception as e:
        return {"exists": False, "error": str(e)}

@app.post("/api/download_data")
async def download_data(req: DownloadRequest, background_tasks: BackgroundTasks):
    task_id = f"DL-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
    background_tasks.add_task(
        run_batch_download_task_background,
        task_id, req.exchange, req.symbols, req.timeframes, req.start_date, req.end_date, req.trade_types, req.download_mode
    )
    return {"task_id": task_id}

async def run_batch_download_task_background(task_id: str, exchange: str, symbols: list[str], timeframes: list[str], start_date: str, end_date: str, trade_types: list[str], download_mode: str):
    from core.data_feed.batch_downloader import SmartBatchDownloader
    import itertools
    
    total_tasks = len(symbols) * len(timeframes) * len(trade_types)
    current_task = 0

    def get_progress_callback(symbol_name, task_index):
        def progress_callback(progress, message):
            # Scale progress across all tasks
            overall_progress = int(((task_index + (progress / 100)) / total_tasks) * 100)
            
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(manager.broadcast({
                    "type": "download_progress",
                    "task_id": task_id,
                    "symbol": symbol_name,
                    "progress": overall_progress,
                    "message": f"[{task_index + 1}/{total_tasks}] {symbol_name}: {message}"
                }))
            except RuntimeError:
                pass
        return progress_callback

    for symbol, timeframe, trade_type in itertools.product(symbols, timeframes, trade_types):
        try:
            actual_start = start_date
            actual_end = end_date
            
            if download_mode == "fill_gap":
                coverage = await get_data_coverage(exchange, symbol, timeframe, trade_type)
                if coverage.get("exists"):
                    local_start = coverage["start_date"].split(" ")[0]
                    local_end = coverage["end_date"].split(" ")[0]
                    if actual_end < local_start:
                        actual_end = local_start
                    elif actual_start > local_end:
                        actual_start = local_end
            
            downloader = SmartBatchDownloader()
            downloader.exchange_id = exchange # We need a way to pass exchange to it!
            callback = get_progress_callback(symbol, current_task)
            callback(0, "Initializing...")
            await asyncio.sleep(0.5) # Slight delay for UI
            
            result = await asyncio.to_thread(
                downloader.download_ohlcv_batch,
                exchange_id=exchange,
                symbol=symbol,
                timeframe=timeframe,
                start_date=actual_start,
                end_date=actual_end,
                trade_type=trade_type,
                progress_callback=callback,
                download_mode=download_mode
            )
            
            if isinstance(result, dict) and not result.get('success', True):
                callback(100, f"Error: {result.get('error', 'Unknown error')}")
            else:
                callback(100, "Successfully completed!")
                
        except Exception as e:
            callback = get_progress_callback(symbol, current_task)
            callback(100, f"Error: {str(e)}")
            
        current_task += 1

@app.websocket("/ws/monitor")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            # Just keep the connection alive
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
