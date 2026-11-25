"""Isolated CLI/Web execution tests for review items 7, 8 and 14."""

import asyncio
import copy
import json
import logging
import threading
from argparse import Namespace
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from core.commands.mine import run_miner
from core.data_feed.naming import data_path
from core.data_feed.real_client import RealDataClient
from core.miner.expressions import FactorExpressionAST
from core.startup_validation import StartupValidationError, validate_mining_startup
from core.utils.dynamic_loader import ModuleLoadReport


PAIRS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]


def config(mode="sequential_single"):
    return {"paradigm": "GP", "max_iterations": 2,
            "data_feeds": {"required_streams": ["close"], "pairs": PAIRS,
                           "exchange": "binance", "instrument_type": "futures", "timeframe": "1m",
                           "mine_period": [["2024-01-01", "2024-01-01 01:19:00"]], "mining_mode": mode}}


@pytest.mark.parametrize("configured,override,expected", [(2, None, 2), (2, 3, 3), (None, None, 5)])
def test_cross_asset_cli_resolves_optional_iterations(tmp_path, monkeypatch, configured, override, expected):
    cfg = config("cross_asset")
    if configured is None:
        cfg.pop("max_iterations")
    else:
        cfg["max_iterations"] = configured
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    director = Mock()
    director.run.return_value = []
    director.get_latest_factor_ids.return_value = []
    factory = Mock(return_value=director)
    monkeypatch.setattr("core.commands.mine.load_user_modules", lambda _: ModuleLoadReport())
    monkeypatch.setattr("core.commands.mine.FactorMinerDirector", factory)
    monkeypatch.setattr("core.data_feed.real_client.RealDataClient", lambda cfg: object())
    run_miner(Namespace(config=str(path), miner="GP", iterations=override, user_dir=str(tmp_path)))
    director.run.assert_called_once_with(max_iterations=expected)
    assert factory.call_args.args[0]["data_feeds"]["pairs"] == PAIRS


def write_market_data(root):
    for pair, base in zip(PAIRS, [100., 200.]):
        path = data_path(root / "data", "binance", pair, "1m", "futures")
        path.parent.mkdir(parents=True, exist_ok=True)
        close = base + np.arange(80)
        pd.DataFrame({"date": pd.date_range("2024-01-01", periods=80, freq="min"),
                      "open": close, "high": close + 1, "low": close - 1,
                      "close": close, "volume": 10.}).to_feather(path)


@pytest.mark.parametrize("mode,produce,expected_status", [
    (None, [True, True], "completed"),
    ("sequential_single", [True, True], "completed"),
    ("sequential_single", [False, True], "completed"),
    ("sequential_single", [False, False], "completed_empty"),
    ("cross_asset", [True], "completed"),
])
def test_web_runs_all_pairs_and_aggregates_results(tmp_path, monkeypatch, mode, produce, expected_status):
    import api.main as api

    monkeypatch.chdir(tmp_path)
    write_market_data(tmp_path)
    config_folder = tmp_path / "user_workspace" / "configs"
    config_folder.mkdir(parents=True)
    cfg = config(mode)
    if mode is None:
        cfg["data_feeds"].pop("mining_mode")
    path = config_folder / "test.json"
    path.write_text(json.dumps(cfg))
    calls, events = [], []
    class Director:
        def __init__(self, run_config, client):
            self.index = len(calls)
            self.factor = FactorExpressionAST("close")
            self.factor.logic_hash = f"hash_{self.index}"
            self.factor.metrics = {"IC": .1}
            calls.append((copy.deepcopy(run_config), client.get_data()))
        def run(self, max_iterations, progress_callback):
            assert max_iterations == 2
            for epoch in [1, 2]:
                progress_callback(epoch, 2, self.factor)
            return [self.factor] if produce[self.index] else []
        def get_latest_factor_ids(self):
            return [f"fac_{self.index}"] if produce[self.index] else []
    async def broadcast(event):
        events.append(copy.deepcopy(event))
    monkeypatch.setattr("core.miner.director.FactorMinerDirector", Director)
    monkeypatch.setattr("core.utils.dynamic_loader.load_user_modules", lambda _: ModuleLoadReport())
    monkeypatch.setattr(api.manager, "broadcast", broadcast)
    monkeypatch.setattr(api.TaskManager, "tasks", {"test": {"logs": [], "status": "running"}})
    async def run():
        await api.run_mining_task_background("test", "GP", "test.json")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    asyncio.run(run())
    task = api.TaskManager.tasks["test"]
    assert task["status"] == expected_status, task.get("error_msg")
    assert task["result_count"] == sum(produce)
    assert all(event["complexity"] == "Depth: 1" for event in events if event["type"] == "scatter")
    assert [factor["factor_id"] for factor in task["factors"]] == [f"fac_{i}" for i, present in enumerate(produce) if present]
    if mode in (None, "sequential_single"):
        assert [call[0]["data_feeds"]["pairs"] for call in calls] == [[pair] for pair in PAIRS]
        assert [call[1].close.iloc[0] for call in calls] == [100., 200.]
        progress = [event["task"]["progress"] for event in events if event["type"] == "task_update"]
        assert progress[:4] == [25, 50, 75, 100]
    else:
        assert len(calls) == 1
        assert calls[0][0]["data_feeds"]["pairs"] == PAIRS
        assert list(calls[0][1]["close"].columns) == PAIRS
    assert json.loads(path.read_text()) == cfg


def test_startup_aggregates_invalid_mining_mode_with_other_errors():
    cfg = config("cross_assets")
    cfg["max_iterations"] = 0
    cfg["data_feeds"]["required_streams"] = []
    report = ModuleLoadReport(errors=["synthetic plugin load failure"])
    with pytest.raises(StartupValidationError) as error:
        validate_mining_startup(cfg, report)
    for message in ["data_feeds.mining_mode", "cross_assets", "max_iterations",
                    "required_streams", "synthetic plugin load failure"]:
        assert message in str(error.value)


@pytest.mark.parametrize("mode", ["cross_assets", "sequential_singe", "", None, 7, ["cross_asset"]])
def test_reader_rejects_invalid_mining_mode_before_loading(monkeypatch, mode):
    load = Mock()
    monkeypatch.setattr(RealDataClient, "_load_all_data", load)
    with pytest.raises(ValueError, match="data_feeds.mining_mode"):
        RealDataClient(config(mode))
    load.assert_not_called()


def test_cli_rejects_unknown_mining_mode_before_data_or_director(tmp_path, monkeypatch, caplog):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config("cross_assets")))
    reader = Mock(return_value=object())
    director = Mock()
    director.run.return_value = []
    director.get_latest_factor_ids.return_value = []
    factory = Mock(return_value=director)
    monkeypatch.setattr("core.commands.mine.load_user_modules", lambda _: ModuleLoadReport())
    monkeypatch.setattr("core.data_feed.real_client.RealDataClient", reader)
    monkeypatch.setattr("core.commands.mine.FactorMinerDirector", factory)
    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as error:
        run_miner(Namespace(config=str(path), miner="GP", iterations=None, user_dir=str(tmp_path)))
    assert error.value.code == 1
    assert "data_feeds.mining_mode" in caplog.text
    assert "cross_assets" in caplog.text
    reader.assert_not_called()
    factory.assert_not_called()
    assert not (tmp_path / "factor_db").exists()


def test_api_rejects_unknown_mining_mode_before_data_or_director(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    import api.main as api

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "user_workspace" / "configs"
    folder.mkdir(parents=True)
    (folder / "test.json").write_text(json.dumps(config("cross_assets")))
    reader = Mock(return_value=object())
    director = Mock()
    director.run.return_value = []
    director.get_latest_factor_ids.return_value = []
    factory = Mock(return_value=director)
    events = []
    async def broadcast(event):
        events.append(copy.deepcopy(event))
    monkeypatch.setattr("core.utils.dynamic_loader.load_user_modules", lambda _: ModuleLoadReport())
    monkeypatch.setattr("core.data_feed.real_client.RealDataClient", reader)
    monkeypatch.setattr("core.miner.director.FactorMinerDirector", factory)
    monkeypatch.setattr(api.TaskManager, "tasks", {})
    monkeypatch.setattr(api.manager, "broadcast", broadcast)
    with TestClient(api.app) as client:
        response = client.post("/api/launch", json={"miner": "GP", "config": "test.json"})
    assert response.status_code == 200  # The existing API validates its queued job.
    task = api.TaskManager.tasks[response.json()["task_id"]]
    assert task["status"] == "failed"
    assert task["error_msg"] == "Mining task failed. Check the selected configuration and local server logs."
    assert "Traceback" not in task["error_msg"]
    assert task["result_count"] == 0 and task["factors"] == []
    assert events[-1]["task"]["status"] == "failed"
    reader.assert_not_called()
    factory.assert_not_called()
    assert not (tmp_path / "factor_db").exists()


@pytest.mark.parametrize("mode", [None, "sequential_single", "cross_asset"])
def test_valid_mining_modes_preserve_reader_and_cli_routing(tmp_path, monkeypatch, mode):
    from core.miner.director import FactorMinerDirector

    monkeypatch.chdir(tmp_path)
    write_market_data(tmp_path)
    cfg = config(mode)
    if mode is None:
        cfg["data_feeds"].pop("mining_mode")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg))
    validate_mining_startup(cfg)
    direct = RealDataClient(cfg)
    assert direct.mining_mode == (mode or "sequential_single")
    calls = []
    class Director:
        def __init__(self, run_config, client):
            self.config = run_config
            self.data_client = client
            lineage = FactorMinerDirector._snapshot_lineage(self)
            calls.append((client.get_data(), client.get_returns(), lineage))
        def run(self, max_iterations):
            return []
        def get_latest_factor_ids(self):
            return []
    monkeypatch.setattr("core.commands.mine.load_user_modules", lambda _: ModuleLoadReport())
    monkeypatch.setattr("core.commands.mine.FactorMinerDirector", Director)
    run_miner(Namespace(config=str(path), miner="GP", iterations=None, user_dir=str(tmp_path)))
    if mode == "cross_asset":
        assert direct.get_data()["close"].shape == (80, 2)
        assert list(direct.get_data()["close"].columns) == PAIRS
        assert len(calls) == 1
        assert calls[0][0]["close"].shape == (80, 2)
        assert calls[0][2]["pairs"] == PAIRS
    else:
        assert direct.get_data().close.iloc[0] == 100.
        assert len(calls) == 2
        assert [data.close.iloc[0] for data, _, _ in calls] == [100., 200.]
        assert [lineage["pairs"] for _, _, lineage in calls] == [[pair] for pair in PAIRS]
    for data, labels, lineage in calls:
        assert lineage["mining_mode"] == (mode or "sequential_single")
        close = data["close"]
        expected = close.shift(-1).div(close).sub(1.)
        if isinstance(labels, pd.DataFrame):
            pd.testing.assert_frame_equal(labels, expected)
        else:
            pd.testing.assert_series_equal(labels, expected.rename("returns"))


@pytest.mark.parametrize("success", [True, False])
def test_worker_download_progress_reaches_main_loop_and_preserves_result(monkeypatch, success):
    import api.main as api

    events, workers = [], []
    main_thread = threading.get_ident()
    def download(**kwargs):
        workers.append(threading.get_ident())
        kwargs["progress_callback"](37, "worker progress marker")
        return {"success": success, "error": "fixture failure"}
    async def broadcast(event):
        assert threading.get_ident() == main_thread
        events.append(event)
    monkeypatch.setattr("core.data_feed.batch_downloader.SmartBatchDownloader.download_ohlcv_batch",
                        lambda self, **kwargs: download(**kwargs))
    monkeypatch.setattr(api.manager, "broadcast", broadcast)
    async def run():
        await api.run_batch_download_task_background("download", "okx", PAIRS, ["1m"],
            "2024-01-01", "2024-01-02", ["futures"], "merge")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
    asyncio.run(run())
    assert all(worker != main_thread for worker in workers)
    intermediate = [event for event in events if "worker progress marker" in event["message"]]
    assert [event["progress"] for event in intermediate] == [18, 68]
    assert [event["symbol"] for event in intermediate] == PAIRS
    terminal = "Successfully completed!" if success else "Download reported a problem. See local server logs."
    assert sum(terminal in event["message"] for event in events) == 2
    assert events[-1]["progress"] == 100


def test_web_mining_drains_delayed_progress_before_terminal_state(tmp_path, monkeypatch):
    import api.main as api

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "user_workspace" / "configs"
    folder.mkdir(parents=True)
    (folder / "test.json").write_text(json.dumps(config("cross_asset")))
    events = []

    class Director:
        def __init__(self, run_config, client):
            pass

        def run(self, max_iterations, progress_callback):
            progress_callback(1, 2, None)
            return []

        def get_latest_factor_ids(self):
            return []

    monkeypatch.setattr("core.miner.director.FactorMinerDirector", Director)
    monkeypatch.setattr("core.data_feed.real_client.RealDataClient", lambda _: object())
    monkeypatch.setattr("core.utils.dynamic_loader.load_user_modules", lambda _: ModuleLoadReport())
    monkeypatch.setattr(api.TaskManager, "tasks", {"test": {"id": "test", "logs": [], "status": "running"}})

    async def run():
        progress_started = asyncio.Event()
        release_progress = asyncio.Event()

        async def broadcast(event):
            snapshot = copy.deepcopy(event)
            if event["type"] == "task_update" and event["task"]["status"] == "running":
                progress_started.set()
                await release_progress.wait()
            events.append(snapshot)

        monkeypatch.setattr(api.manager, "broadcast", broadcast)
        task = asyncio.create_task(api.run_mining_task_background("test", "GP", "test.json"))
        try:
            await asyncio.wait_for(progress_started.wait(), 2)
            for _ in range(5):
                await asyncio.sleep(0)
            assert not events, "a terminal update overtook pending progress"
        finally:
            release_progress.set()
            await asyncio.wait_for(task, 2)

    asyncio.run(run())
    assert [event["task"]["status"] for event in events] == ["running", "completed_empty"]
    assert [event["task"]["progress"] for event in events] == [50, 100]


def test_parallel_web_mining_keeps_logs_and_logger_lifetimes_separate(tmp_path, monkeypatch):
    import api.main as api

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "user_workspace" / "configs"
    folder.mkdir(parents=True)
    for name in ("A", "B"):
        (folder / f"{name}.json").write_text(json.dumps({**config("cross_asset"), "probe_name": name}))
    both_started = threading.Barrier(2, timeout=2)
    first_finished = threading.Event()
    logger = logging.getLogger("core.task_log_probe")
    logger_names = ("core", "custom_miners", "custom_fitness")
    for name in logger_names:
        monkeypatch.setattr(logging.getLogger(name), "level", logging.WARNING)

    class Director:
        def __init__(self, run_config, client):
            self.name = run_config["probe_name"]
            logger.info("INIT_%s", self.name)

        def run(self, max_iterations, progress_callback):
            both_started.wait()
            if self.name == "B":
                assert first_finished.wait(2)
            logger.info("ONLY_%s", self.name)
            # Candidate evaluation introduces a second thread boundary. Its
            # logs must retain the outer mining task's identity as well.
            from core.evaluation.evaluator import ParallelEvaluator
            from types import SimpleNamespace
            frame = pd.DataFrame({"close": [1., 2., 3.]})
            def compute(data):
                logger.info("EVALUATOR_%s", self.name)
                return data["close"]
            feedback = ParallelEvaluator(None, {}).evaluate_on(
                [SimpleNamespace(compute=compute)], frame, pd.Series([.1, .2, .3])
            )
            assert feedback.results[0].succeeded
            return []

        def get_latest_factor_ids(self):
            return []

    async def broadcast(event):
        await asyncio.sleep(0)

    monkeypatch.setattr("core.miner.director.FactorMinerDirector", Director)
    monkeypatch.setattr("core.data_feed.real_client.RealDataClient", lambda _: object())
    monkeypatch.setattr("core.utils.dynamic_loader.load_user_modules", lambda _: ModuleLoadReport())
    monkeypatch.setattr(api.manager, "broadcast", broadcast)
    monkeypatch.setattr(api.TaskManager, "tasks", {
        name: {"id": name, "logs": [], "status": "running"} for name in ("A", "B")
    })

    async def run():
        first = asyncio.create_task(api.run_mining_task_background("A", "GP", "A.json"))
        second = asyncio.create_task(api.run_mining_task_background("B", "GP", "B.json"))
        try:
            await asyncio.wait_for(first, 3)
            # B still needs INFO logs after A has removed its own handler.
            assert all(logging.getLogger(name).level == logging.INFO for name in logger_names)
            logger.info("UNSCOPED")
        finally:
            first_finished.set()
            await asyncio.wait_for(second, 3)

    asyncio.run(run())
    for name in ("A", "B"):
        messages = [line.rsplit(" ", 1)[-1] for line in api.TaskManager.tasks[name]["logs"]]
        assert messages == [f"INIT_{name}", f"ONLY_{name}", f"EVALUATOR_{name}"]
        assert api.TaskManager.tasks[name]["status"] == "completed_empty"
    assert all(logging.getLogger(name).level == logging.WARNING for name in logger_names)


def test_web_mining_initialization_and_run_share_worker_without_blocking_loop(tmp_path, monkeypatch):
    import api.main as api

    monkeypatch.chdir(tmp_path)
    write_market_data(tmp_path)
    cfg = config()
    cfg["data_feeds"]["pairs"] = [PAIRS[0]]
    folder = tmp_path / "user_workspace" / "configs"
    folder.mkdir(parents=True)
    (folder / "test.json").write_text(json.dumps(cfg))
    main_thread = threading.get_ident()
    stages, released_during_read, events = [], [], []
    release_read = threading.Event()
    original_read = pd.read_feather

    class Director:
        def __init__(self, run_config, client):
            stages.append(("director_init", threading.get_ident()))
            assert len(client.get_data()) == 80

        def run(self, max_iterations, progress_callback):
            stages.append(("run", threading.get_ident()))
            progress_callback(1, 1, None)
            return []

        def get_latest_factor_ids(self):
            stages.append(("factor_ids", threading.get_ident()))
            return []

    async def broadcast(event):
        events.append(copy.deepcopy(event))

    monkeypatch.setattr("core.miner.director.FactorMinerDirector", Director)
    monkeypatch.setattr("core.utils.dynamic_loader.load_user_modules", lambda _: ModuleLoadReport())
    monkeypatch.setattr(api.TaskManager, "tasks", {"test": {"id": "test", "logs": [], "status": "running"}})
    monkeypatch.setattr(api.manager, "broadcast", broadcast)

    async def run():
        loop = asyncio.get_running_loop()
        read_started = asyncio.Event()
        def read(*args, **kwargs):
            stages.append(("read_feather", threading.get_ident()))
            loop.call_soon_threadsafe(read_started.set)
            # A responsive loop must release the real file read. The timeout
            # only bounds the original broken implementation's blocking wait.
            released_during_read.append(release_read.wait(1))
            return original_read(*args, **kwargs)
        monkeypatch.setattr(pd, "read_feather", read)
        async def release_from_loop():
            await read_started.wait()
            assert (await api.get_tasks())["tasks"][0]["status"] == "running"
            release_read.set()
        api.TaskManager.tasks["test"]["start_time"] = "2026-10-31T00:00:00"
        heartbeat = asyncio.create_task(release_from_loop())
        try:
            await api.run_mining_task_background("test", "GP", "test.json")
            await asyncio.wait_for(heartbeat, 2)
        finally:
            release_read.set()
            if not heartbeat.done():
                heartbeat.cancel()

    asyncio.run(run())
    assert released_during_read == [True], "data initialization blocked the API event loop"
    assert [name for name, _ in stages] == ["read_feather", "director_init", "run", "factor_ids"]
    assert len({thread for _, thread in stages}) == 1
    assert stages[0][1] != main_thread
    assert api.TaskManager.tasks["test"]["status"] == "completed_empty"
    assert events[-1]["task"]["status"] == "completed_empty"


def test_config_preview_exposes_only_typed_launchpad_fields(tmp_path, monkeypatch):
    import api.main as api

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "user_workspace" / "configs"
    folder.mkdir(parents=True)
    private = "synthetic-private-config-value"
    selected = {
        **config(), "population_size": 12,
        "llm_api_config": {"keys": [private], "nested": {"password": private}},
        "data_feeds": {**config()["data_feeds"], "credentials": {"secret": private}},
        "fitness": {"hook": "example_hook", "parameters": {"secret": private}},
        "search_space": {"allowed_operators": ["add", "sub"], "private": private},
    }
    path = folder / "selected.json"
    path.write_text(json.dumps(selected))
    invalid_types = {
        "population_size": {"secret": private}, "max_iterations": True,
        "data_feeds": {"timeframe": {"secret": private}, "mining_mode": [private],
                       "pairs": [{"secret": private}], "mine_period": [["2024-01-01", {"secret": private}]]},
        "fitness": {"hook": {"secret": private}},
        "search_space": {"allowed_operators": [{"secret": private}]},
    }
    (folder / "invalid-types.json").write_text(json.dumps(invalid_types))
    (folder / "invalid-json.json").write_text('{"api_key": "synthetic-value",')
    summaries = asyncio.run(api.get_configs())["configs"]
    assert summaries["selected.json"] == {
        "population_size": 12, "max_iterations": 2,
        "data_feeds": {name: selected["data_feeds"][name] for name in ("pairs", "timeframe", "mining_mode", "mine_period")},
        "fitness": {"hook": "example_hook"},
        "search_space": {"allowed_operators": ["add", "sub"]},
    }
    assert summaries["invalid-types.json"] == {"data_feeds": {}}
    assert summaries["invalid-json.json"] == {"error": "Invalid or unavailable JSON configuration."}
    assert private not in json.dumps(summaries)
    assert json.loads(path.read_text()) == selected


def test_config_selection_requires_a_regular_listed_json_file(tmp_path, monkeypatch):
    from fastapi import BackgroundTasks, HTTPException
    import api.main as api

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "user_workspace" / "configs"
    folder.mkdir(parents=True)
    (folder / "selected.json").write_text(json.dumps(config()))
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(config()))
    (folder / "linked.json").symlink_to(outside)
    (folder / "internal-link.json").symlink_to(folder / "selected.json")
    (folder / "directory.json").mkdir()
    (folder / "notes.txt").write_text("fixture")
    monkeypatch.setattr(api.TaskManager, "tasks", {})
    async def broadcast(event):
        pass
    monkeypatch.setattr(api.manager, "broadcast", broadcast)

    async def run():
        assert list((await api.get_configs())["configs"]) == ["selected.json"]
        for name in (str(outside), "nested/selected.json", "linked.json", "internal-link.json", "directory.json", "notes.txt", "missing.json"):
            with pytest.raises(HTTPException) as error:
                await api.launch_mining(api.LaunchRequest(miner="GP", config=name), BackgroundTasks())
            assert error.value.status_code == 422
        assert api.TaskManager.tasks == {}
        jobs = BackgroundTasks()
        accepted = await api.launch_mining(api.LaunchRequest(miner="GP", config="selected.json"), jobs)
        assert accepted["task_id"] in api.TaskManager.tasks
        assert len(jobs.tasks) == 1
    asyncio.run(run())


def test_local_browser_cors_and_websocket_origins(monkeypatch):
    from fastapi.testclient import TestClient
    from starlette.websockets import WebSocketDisconnect
    from types import SimpleNamespace
    import api.main as api

    monkeypatch.setattr(api.TaskManager, "tasks", {})
    monkeypatch.setattr(api.TaskManager, "download_tasks", {})
    monkeypatch.setattr(api.manager, "active_connections", [])
    with TestClient(api.app) as client:
        for origin in api.LOCAL_BROWSER_ORIGINS:
            response = client.get("/api/health", headers={"Origin": origin})
            assert response.headers["access-control-allow-origin"] == origin
            with client.websocket_connect("/ws/monitor", headers={"Origin": origin}):
                pass
        for origin in ("https://example.invalid", "null"):
            response = client.options("/api/launch", headers={
                "Origin": origin, "Access-Control-Request-Method": "POST",
            })
            assert response.status_code == 400
            assert "access-control-allow-origin" not in response.headers
            with pytest.raises(WebSocketDisconnect) as error:
                with client.websocket_connect("/ws/monitor", headers={"Origin": origin}):
                    pass
            assert error.value.code == 1008
        with client.websocket_connect("/ws/monitor"):
            pass
    assert api.manager.active_connections == []
    assert api.websocket_origin_allowed(SimpleNamespace(headers={}, client=SimpleNamespace(host="127.0.0.1")))
    assert api.websocket_origin_allowed(SimpleNamespace(headers={}, client=SimpleNamespace(host="::1")))
    assert not api.websocket_origin_allowed(SimpleNamespace(headers={}, client=SimpleNamespace(host="192.0.2.1")))


def test_mining_ui_logs_and_failure_state_do_not_expose_credentials_or_tracebacks(tmp_path, monkeypatch):
    import api.main as api

    monkeypatch.chdir(tmp_path)
    folder = tmp_path / "user_workspace" / "configs"
    folder.mkdir(parents=True)
    private = "synthetic-private-config-value"
    environment_private = "synthetic-private-environment-value"
    cfg = {**config("cross_asset"), "llm_api_config": {"keys": [private]}}
    (folder / "selected.json").write_text(json.dumps(cfg))
    monkeypatch.setenv("FM_TEST_API_KEY", environment_private)
    events = []
    logger = logging.getLogger("core.synthetic_public_diagnostics")
    class Director:
        def __init__(self, run_config, client):
            pass
        def run(self, max_iterations, progress_callback):
            logger.info("Progress fixture: %s / %s", private, environment_private)
            logger.warning("opaque-warning-fixture")
            logger.info("Exception fixture: %s", RuntimeError("opaque-exception-argument"))
            logger.info("Mapped exception: %(error)s", {"error": RuntimeError("opaque-mapped-exception")})
            try:
                raise RuntimeError("opaque-exception-body")
            except RuntimeError:
                logger.exception("Exception log fixture")
                raise
        def get_latest_factor_ids(self):
            return []
    async def broadcast(event):
        events.append(copy.deepcopy(event))
    monkeypatch.setattr("core.miner.director.FactorMinerDirector", Director)
    monkeypatch.setattr("core.data_feed.real_client.RealDataClient", lambda _: object())
    monkeypatch.setattr("core.utils.dynamic_loader.load_user_modules", lambda _: ModuleLoadReport())
    monkeypatch.setattr(api.manager, "broadcast", broadcast)
    monkeypatch.setattr(api.TaskManager, "tasks", {"fixture": {"id": "fixture", "logs": [], "status": "running", "start_time": "2026-11-25"}})
    asyncio.run(api.run_mining_task_background("fixture", "GP", "selected.json"))
    task = api.TaskManager.tasks["fixture"]
    assert task["status"] == "failed"
    assert any("Progress fixture" in message for message in task["logs"])
    serialized = json.dumps({"events": events, "task": task})
    for protected in (private, environment_private, "opaque-warning-fixture", "opaque-exception-argument", "opaque-mapped-exception", "opaque-exception-body", "Traceback"):
        assert protected not in serialized
    assert "[REDACTED]" in serialized


@pytest.mark.parametrize("candidate,expected", [
    (FactorExpressionAST({"op": "add", "left": "close", "right": "volume"}), "Depth: 2"),
    (None, None),
])
def test_scatter_uses_candidate_complexity_without_fabricating_values(candidate, expected):
    from types import SimpleNamespace
    import api.main as api

    assert api._candidate_complexity(candidate) == expected
    assert api._candidate_complexity(SimpleNamespace(complexity=7)) == 7
    assert api._candidate_complexity(SimpleNamespace()) is None
