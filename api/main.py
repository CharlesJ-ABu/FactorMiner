from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from api.ws_manager import manager
import asyncio
import random
import time
import os
import datetime
import traceback
import dataclasses
from pathlib import Path

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

class LifecycleUpdateRequest(BaseModel):
    lifecycle_status: str

class BatchLifecycleUpdateRequest(BaseModel):
    factor_ids: list[str]
    lifecycle_status: str

class CompareFactorsRequest(BaseModel):
    factor_ids: list[str]

LIFECYCLE_STATUSES = {"DISCOVERED", "INSPECTED", "PAPER_TRADING", "LIVE", "RETIRED"}

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
    load_report = load_user_modules("user_workspace")
    
    # Only return registered custom miners
    custom_miners = list(MinerRegistry._registry.keys())
    
    return {"miners": custom_miners, "load_errors": load_report.errors}

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

    fallback_symbols = [
        "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT", "ADA/USDT",
        "DOGE/USDT", "AVAX/USDT", "LINK/USDT", "MATIC/USDT", "DOT/USDT", "LTC/USDT",
        "BCH/USDT", "TRX/USDT", "UNI/USDT", "ATOM/USDT", "ETC/USDT", "TON/USDT",
        "NEAR/USDT", "APT/USDT", "ARB/USDT", "OP/USDT", "SUI/USDT", "SEI/USDT",
        "TIA/USDT", "INJ/USDT", "FIL/USDT", "LDO/USDT", "RNDR/USDT", "STX/USDT",
        "ORDI/USDT", "PEPE/USDT", "SHIB/USDT", "WLD/USDT", "GALA/USDT", "FTM/USDT",
    ]
    if trade_type == "futures":
        fallback_symbols = [f"{symbol}:USDT" for symbol in fallback_symbols]
    
    meta = {
        "symbols": fallback_symbols,
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
                market_suffix = "/USDT:USDT" if trade_type == "futures" else "/USDT"
                usdt_markets = [s for s in ex_instance.markets.keys() if s.endswith(market_suffix)]
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

def _ast_display(node):
    if not isinstance(node, dict) or "op" not in node:
        return str(node)
    return f"{node['op']}({_ast_display(node.get('left'))}, {_ast_display(node.get('right'))})"

def _factor_logic(metadata):
    """Return an Inspector-safe, paradigm-aware view of a persisted factor."""
    logic_ref = metadata.logic_reference or {}
    logic_type = logic_ref.get("type")

    if logic_type == "json_ast":
        ast = logic_ref.get("ast", {})
        return {"kind": "ast", "ast": ast, "display": _ast_display(ast)}

    if logic_type == "python_source":
        source_file = logic_ref.get("source_file", "")
        source_path = Path("factor_db") / "sources" / source_file
        source = None
        if source_file and source_path.is_file():
            source = source_path.read_text(encoding="utf-8")
        return {
            "kind": "source",
            "source_file": source_file,
            "source": source,
            "reflection": logic_ref.get("reflection", ""),
        }

    if logic_type == "rl_actions":
        return {
            "kind": "actions",
            "actions": logic_ref.get("actions", []),
            "weights_file": logic_ref.get("weights_file"),
        }

    if logic_type in {"nn_channel", "dl_channel"}:
        model_version = logic_ref.get("model_version")
        model_file = logic_ref.get("model_file")
        if model_file:
            artifact_file = Path(model_file).name
            artifact_path = Path("factor_db") / "models" / artifact_file
        else:
            artifact_file = f"{model_version}.pt" if model_version else None
            artifact_path = (
                Path("factor_db") / "weights" / artifact_file
                if artifact_file
                else None
            )
        return {
            "kind": "nn_channel",
            "model_version": model_version,
            "channel": logic_ref.get("channel"),
            "weights_file": artifact_file,
            "weights_available": bool(artifact_path and artifact_path.is_file()),
            "model_format": logic_ref.get("model_format", "legacy_raw_weights"),
            "features": logic_ref.get("features", []),
            "schema_version": logic_ref.get("schema_version"),
        }

    return {"kind": "unknown", "reference": logic_ref}

def _factor_summary(metadata):
    logic = _factor_logic(metadata)
    if logic["kind"] == "ast":
        display = logic["display"]
    elif logic["kind"] == "source":
        display = logic.get("source_file") or "Python source unavailable"
    elif logic["kind"] == "actions":
        display = " → ".join(logic.get("actions") or []) or "Action trajectory"
    elif logic["kind"] == "nn_channel":
        display = f"NNModel(v={logic.get('model_version')}) [Ch: {logic.get('channel')}]"
    else:
        display = "Stored factor reference"
    return {
        "factor_id": metadata.factor_id,
        "miner_type": metadata.miner_type,
        "lifecycle_status": metadata.lifecycle_status,
        "logic_hash": metadata.logic_hash,
        "metrics": metadata.metrics,
        "created_at": metadata.created_at,
        "display": display,
        "logic_kind": logic["kind"],
        "snapshot_available": bool(metadata.data_lineage.get("snapshot_file")),
    }

def _factor_detail(metadata):
    values_path = Path("factor_db") / "values" / f"{metadata.factor_id}.parquet"
    snapshot_available = values_path.is_file()
    return {
        "metadata": dataclasses.asdict(metadata),
        "logic": _factor_logic(metadata),
        "audit_snapshot": {
            "values_available": snapshot_available,
            "lineage": metadata.data_lineage,
            "message": (
                f"Snapshot contains {metadata.data_lineage.get('snapshot_rows', 0)} aligned factor/forward-return observations."
                if snapshot_available
                else "No factor-value snapshot was saved. Re-run mining to create a real Tearsheet snapshot."
            ),
        },
    }

def _factor_analysis_payload(factor_id: str):
    from core.analysis.factor_analytics import SnapshotAnalysisError, analyze_factor_snapshot
    from core.storage.factor_storage import get_global_storage

    storage = get_global_storage()
    metadata = storage.get_metadata(factor_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Factor {factor_id} was not found")
    snapshot = storage.load_factor_values(factor_id)
    if snapshot is None or snapshot.empty:
        raise HTTPException(
            status_code=409,
            detail="This factor has no persisted value/forward-return snapshot. Re-run mining to generate a real Tearsheet.",
        )
    try:
        analysis = analyze_factor_snapshot(snapshot)
    except SnapshotAnalysisError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"factor": _factor_summary(metadata), "lineage": metadata.data_lineage, "analysis": analysis}

@app.get("/api/dashboard")
async def get_dashboard():
    """Return a compact, storage-backed snapshot for the research dashboard."""
    from core.storage.factor_storage import get_global_storage

    tasks = list(TaskManager.tasks.values())
    recent_tasks = sorted(tasks, key=lambda task: task["start_time"], reverse=True)[:6]
    task_statuses = {status: 0 for status in ("running", "completed", "completed_empty", "failed")}
    for task in tasks:
        task_statuses[task.get("status", "failed")] = task_statuses.get(task.get("status", "failed"), 0) + 1

    records = get_global_storage().list_metadata()
    factors = [_factor_summary(record) for record in records]
    factors_by_miner = {}
    factors_by_lifecycle = {status: 0 for status in LIFECYCLE_STATUSES}
    for factor in factors:
        miner_type = factor["miner_type"]
        lifecycle_status = factor["lifecycle_status"]
        factors_by_miner[miner_type] = factors_by_miner.get(miner_type, 0) + 1
        factors_by_lifecycle[lifecycle_status] = factors_by_lifecycle.get(lifecycle_status, 0) + 1

    completed_tasks = task_statuses["completed"]
    total_tasks = len(tasks)
    top_by_fitness = sorted(
        factors,
        key=lambda factor: factor["metrics"].get("fitness_score", float("-inf")),
        reverse=True,
    )[:5]
    top_by_ic = max(factors, key=lambda factor: factor["metrics"].get("IC", float("-inf")), default=None)

    return {
        "engine_online": True,
        "generated_at": datetime.datetime.now().isoformat(),
        "tasks": {
            "total": total_tasks,
            "statuses": task_statuses,
            "success_rate": (completed_tasks / total_tasks) if total_tasks else None,
            "recent": recent_tasks,
        },
        "factors": {
            "total": len(factors),
            "by_miner": factors_by_miner,
            "by_lifecycle": factors_by_lifecycle,
            "reviewed": sum(factors_by_lifecycle.get(status, 0) for status in ("INSPECTED", "PAPER_TRADING", "LIVE")),
            "top_by_fitness": top_by_fitness,
            "top_by_ic": top_by_ic,
        },
    }

@app.get("/api/factors")
async def get_factors(
    miner: str | None = None,
    lifecycle: str | None = None,
    query: str | None = None,
    sort_by: str = "created_at",
    limit: int = 200,
):
    """List persisted factors for the Inspector; task memory is not used."""
    from core.storage.factor_storage import get_global_storage

    storage = get_global_storage()
    records = storage.list_metadata()
    if miner:
        records = [record for record in records if record.miner_type == miner]
    if lifecycle:
        records = [record for record in records if record.lifecycle_status == lifecycle]

    summaries = [_factor_summary(record) for record in records]
    if query:
        query_lower = query.lower()
        summaries = [
            summary for summary in summaries
            if query_lower in " ".join(
                str(summary.get(key, "")) for key in ("factor_id", "miner_type", "logic_hash", "display")
            ).lower()
        ]

    sorters = {
        "fitness": lambda item: item["metrics"].get("fitness_score", float("-inf")),
        "ic": lambda item: item["metrics"].get("IC", float("-inf")),
        "created_at": lambda item: item.get("created_at", ""),
    }
    summaries.sort(key=sorters.get(sort_by, sorters["created_at"]), reverse=True)
    return {"factors": summaries[:max(1, min(limit, 500))], "total": len(summaries)}

@app.get("/api/factors/{factor_id}/analysis")
async def get_factor_analysis(factor_id: str):
    return _factor_analysis_payload(factor_id)

@app.post("/api/factors/compare")
async def compare_factors(req: CompareFactorsRequest):
    factor_ids = list(dict.fromkeys(req.factor_ids))
    if not 2 <= len(factor_ids) <= 5:
        raise HTTPException(status_code=422, detail="Select between 2 and 5 unique factors for comparison.")
    return {"factors": [_factor_analysis_payload(factor_id) for factor_id in factor_ids]}

@app.patch("/api/factors/lifecycle/batch")
async def batch_update_factor_lifecycle(req: BatchLifecycleUpdateRequest):
    lifecycle_status = req.lifecycle_status.upper()
    if lifecycle_status not in LIFECYCLE_STATUSES:
        allowed = ", ".join(sorted(LIFECYCLE_STATUSES))
        raise HTTPException(status_code=422, detail=f"Unsupported lifecycle status. Use one of: {allowed}")

    factor_ids = list(dict.fromkeys(req.factor_ids))
    if not factor_ids:
        raise HTTPException(status_code=422, detail="Select at least one factor to update.")

    from core.storage.factor_storage import get_global_storage

    storage = get_global_storage()
    updated = []
    missing = []
    for factor_id in factor_ids:
        metadata = storage.update_lifecycle_status(factor_id, lifecycle_status)
        if metadata:
            updated.append(_factor_summary(metadata))
        else:
            missing.append(factor_id)
    return {"updated": updated, "missing": missing, "lifecycle_status": lifecycle_status}

@app.get("/api/factors/{factor_id}")
async def get_factor(factor_id: str):
    from core.storage.factor_storage import get_global_storage

    metadata = get_global_storage().get_metadata(factor_id)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Factor {factor_id} was not found")
    return _factor_detail(metadata)

@app.patch("/api/factors/{factor_id}/lifecycle")
async def update_factor_lifecycle(factor_id: str, req: LifecycleUpdateRequest):
    lifecycle_status = req.lifecycle_status.upper()
    if lifecycle_status not in LIFECYCLE_STATUSES:
        allowed = ", ".join(sorted(LIFECYCLE_STATUSES))
        raise HTTPException(status_code=422, detail=f"Unsupported lifecycle status. Use one of: {allowed}")

    from core.storage.factor_storage import get_global_storage

    metadata = get_global_storage().update_lifecycle_status(factor_id, lifecycle_status)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Factor {factor_id} was not found")
    return _factor_detail(metadata)

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
        "duration": "0s",
        "result_count": 0,
        "factors": [],
        "logs": [],
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
            task["logs"] = [*task["logs"][-199:], log_entry]
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
    
    # Attach only to FactorMiner and dynamically loaded user modules. Setting the
    # root logger globally would duplicate Uvicorn logs and interfere with parallel tasks.
    task_loggers = [logging.getLogger(name) for name in ("core", "custom_miners", "custom_fitness")]
    previous_logger_levels = {task_logger: task_logger.level for task_logger in task_loggers}
    for task_logger in task_loggers:
        task_logger.setLevel(logging.INFO)
        task_logger.addHandler(ws_handler)
    
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

        # 后台任务与 CLI 使用同一套插件发现流程；不能依赖此前是否访问过
        # /api/miners，否则自定义 Miner 在新启动的服务中不会注册。
        from core.utils.dynamic_loader import load_user_modules
        from core.startup_validation import validate_mining_startup

        load_report = load_user_modules("user_workspace")
        validate_mining_startup(config, load_report)
        task["miner"] = config["paradigm"]
        
        from core.data_feed.real_client import RealDataClient
        data_client = RealDataClient(config)
        
        director = FactorMinerDirector(config, data_client)
        max_iter = config.get("max_iterations", 10)
        
        # We must run director.run in a separate thread to avoid blocking FastAPI
        best_factors = await asyncio.to_thread(
            director.run, max_iterations=max_iter, progress_callback=progress_callback
        )
        
        task["progress"] = 100
        if best_factors:
            task["status"] = "completed"
            task["result_count"] = len(best_factors)
            task["hash"] = getattr(best_factors[0], 'logic_hash', 'N/A')
            task["factors"] = [
                {
                    "factor_id": factor_id,
                    "logic_hash": getattr(candidate, "logic_hash", ""),
                    "logic": candidate.to_display_string(max_length=160),
                    "complexity": candidate.get_complexity(),
                    "metrics": candidate.metrics,
                }
                for factor_id, candidate in zip(director.get_latest_factor_ids(), best_factors)
            ]
        else:
            task["status"] = "completed_empty"
            task["result_count"] = 0
            task["error_msg"] = "任务正常结束，但没有产生可保存的有效因子。请检查执行日志、数据和筛选条件。"
            
    except Exception as e:
        task["status"] = "failed"
        task["error_msg"] = str(e) + "\n" + traceback.format_exc()
        
    finally:
        for task_logger in task_loggers:
            task_logger.removeHandler(ws_handler)
            task_logger.setLevel(previous_logger_levels[task_logger])
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
    from core.data_feed.naming import data_path
    
    try:
        target_file = data_path("data", exchange, symbol, timeframe, trade_type)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if not target_file.parent.exists():
        return {"exists": False, "message": "Directory not found"}
    if not target_file.exists():
        return {"exists": False, "message": "No data found"}
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
    task_id = f"DOWNLOAD-{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}"
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
