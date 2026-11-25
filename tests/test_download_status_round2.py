"""Download results must remain failures at the public CLI and Web boundaries."""

import asyncio
import copy
import json
import logging
import sys
import threading
from types import SimpleNamespace

import pytest
from fastapi import BackgroundTasks, HTTPException, WebSocketDisconnect

from core import cli


PAIRS = ["BTC/USDT:USDT", "ETH/USDT:USDT"]


@pytest.mark.parametrize("outcomes,failed_count", [
    ([True, True], 0), ([False, False], 2), ([True, False], 1),
    ([RuntimeError("transport fixture error"), True], 1), ([None, True], 1),
])
def test_public_download_cli_propagates_all_results(monkeypatch, caplog, outcomes, failed_count):
    calls = []
    main_thread = threading.get_ident()
    def download(self, **kwargs):
        assert threading.get_ident() != main_thread
        calls.append(kwargs)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return None
        return {"success": outcome, "error": "unresolved gap fixture"}
    monkeypatch.setattr("core.data_feed.batch_downloader.SmartBatchDownloader.download_ohlcv_batch", download)
    monkeypatch.setattr(sys, "argv", ["factorminer", "download", "--exchange", "okx", "--symbols", ",".join(PAIRS),
        "--timeframes", "1m", "--type", "futures", "--start", "2024-01-01", "--end", "2024-01-02"])
    caplog.set_level(logging.INFO)
    if failed_count:
        with pytest.raises(SystemExit) as error:
            cli.main()
        assert error.value.code == 1
        assert f"{failed_count}/2 targets failed" in caplog.text
        assert "Batch download completed" not in caplog.text
        assert "Download failed for" in caplog.text
        expected_reason = ("transport fixture error" if isinstance(outcomes[0], Exception)
                           else "no successful result" if outcomes[0] is None else "unresolved gap fixture")
        assert expected_reason in caplog.text
    else:
        assert cli.main() is None
        assert "all 2 targets succeeded" in caplog.text
    assert [call["symbol"] for call in calls] == PAIRS


@pytest.mark.parametrize("outcomes,status,succeeded,failed", [
    ([True, True], "completed", 2, 0), ([False, False], "failed", 0, 2),
    ([True, False], "partial_failed", 1, 1),
    ([RuntimeError("transport fixture error"), True], "partial_failed", 1, 1),
    ([None, None], "failed", 0, 2),
])
def test_web_download_broadcasts_explicit_batch_terminal_state(monkeypatch, outcomes, status, succeeded, failed):
    import api.main as api

    events, calls = [], []
    main_thread = threading.get_ident()
    def download(self, **kwargs):
        assert threading.get_ident() != main_thread
        calls.append(kwargs)
        kwargs["progress_callback"](37, "worker progress fixture")
        # A worker's local 100% progress does not establish batch success.
        kwargs["progress_callback"](100, "worker complete fixture")
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        if outcome is None:
            return None
        return {"success": outcome, "error": "unresolved gap fixture"}
    async def broadcast(event):
        assert threading.get_ident() == main_thread
        # Exercise scheduling so terminal publication must drain worker events.
        await asyncio.sleep(0)
        events.append(event)
    monkeypatch.setattr("core.data_feed.batch_downloader.SmartBatchDownloader.download_ohlcv_batch", download)
    monkeypatch.setattr(api.manager, "broadcast", broadcast)
    asyncio.run(api.run_batch_download_task_background("status-fixture", "okx", PAIRS, ["1m"],
        "2024-01-01", "2024-01-02", ["futures"], "merge"))
    terminal = events[-1]
    assert terminal["status"] == status
    assert terminal["progress"] == 100
    assert terminal["succeeded"] == succeeded
    assert terminal["failed"] == failed
    assert terminal["total"] == 2
    assert all(event["task_id"] == "status-fixture" for event in events)
    assert all(event["status"] == "running" for event in events[:-1])
    assert [result["symbol"] for result in terminal["results"]] == PAIRS
    assert sum(result["success"] for result in terminal["results"]) == succeeded
    assert all(result["error"] for result in terminal["results"] if not result["success"])
    intermediate = [event for event in events if event["message"].endswith("worker progress fixture")]
    assert [event["progress"] for event in intermediate] == [18, 68]
    assert all(event["timeframe"] == "1m" and event["trade_type"] == "futures" for event in intermediate)


def test_download_submission_has_unique_ids_and_rejects_empty_jobs():
    import api.main as api

    request = dict(exchange="okx", symbols=PAIRS, timeframes=["1m"], start_date="2024-01-01", end_date="2024-01-02")
    async def submit():
        first = await api.download_data(api.DownloadRequest(**request), BackgroundTasks())
        second = await api.download_data(api.DownloadRequest(**request), BackgroundTasks())
        assert first["task_id"] != second["task_id"]
        with pytest.raises(HTTPException) as error:
            await api.download_data(api.DownloadRequest(**{**request, "symbols": []}), BackgroundTasks())
        assert error.value.status_code == 422
    asyncio.run(submit())


class ReplaySocket:
    def __init__(self):
        self.messages = []
        self.accepted = False
        self.headers = {}
        self.client = SimpleNamespace(host="testclient")

    async def accept(self):
        self.accepted = True

    async def send_json(self, message):
        self.messages.append(copy.deepcopy(message))

    async def receive_text(self):
        raise WebSocketDisconnect()


def test_web_download_queued_state_is_available_to_a_new_connection(monkeypatch):
    import api.main as api

    monkeypatch.setattr(api.TaskManager, "download_tasks", {}, raising=False)
    monkeypatch.setattr(api.manager, "active_connections", [])
    async def run():
        request = api.DownloadRequest(exchange="okx", symbols=PAIRS, timeframes=["1m"],
            start_date="2024-01-01", end_date="2024-01-02")
        accepted = await api.download_data(request, BackgroundTasks())
        socket = ReplaySocket()
        await api.websocket_endpoint(socket)
        matching = [event for event in socket.messages if event.get("task_id") == accepted["task_id"]]
        assert len(matching) == 1
        assert matching[0]["type"] == "download_progress"
        assert matching[0]["status"] == "running"
        assert matching[0]["progress"] == 0
        assert socket.accepted
        assert api.manager.active_connections == []
    asyncio.run(run())


@pytest.mark.parametrize("outcomes,status", [
    ([True, True], "completed"), ([False, False], "failed"), ([True, False], "partial_failed"),
])
def test_web_download_reconnect_replays_terminal_state(monkeypatch, outcomes, status):
    import api.main as api

    monkeypatch.setattr(api.TaskManager, "download_tasks", {}, raising=False)
    monkeypatch.setattr(api.TaskManager, "tasks", {})
    monkeypatch.setattr(api.manager, "active_connections", [])
    calls = []
    main_thread = threading.get_ident()
    def download(self, **kwargs):
        assert threading.get_ident() != main_thread
        calls.append(kwargs["symbol"])
        kwargs["progress_callback"](60, "offline worker progress")
        return {"success": outcomes[len(calls) - 1], "error": "offline download failure"}
    monkeypatch.setattr("core.data_feed.batch_downloader.SmartBatchDownloader.download_ohlcv_batch", download)

    async def run():
        request = api.DownloadRequest(exchange="okx", symbols=PAIRS, timeframes=["1m"],
            start_date="2024-01-01", end_date="2024-01-02")
        jobs = BackgroundTasks()
        accepted = await api.download_data(request, jobs)
        # The real broadcast manager has no connections while this job finishes.
        await jobs()
        for _ in range(2):
            socket = ReplaySocket()
            await api.websocket_endpoint(socket)
            matching = [event for event in socket.messages if event.get("task_id") == accepted["task_id"]]
            assert len(matching) == 1
            terminal = matching[0]
            assert terminal["type"] == "download_progress"
            assert terminal["status"] == status
            assert terminal["progress"] == 100
            assert terminal["succeeded"] == sum(outcomes)
            assert terminal["failed"] == 2 - sum(outcomes)
            assert [result["symbol"] for result in terminal["results"]] == PAIRS
            assert all(result["error"] for result in terminal["results"] if not result["success"])
            assert api.manager.active_connections == []
        # Download snapshots must not change the mining list/stats task schema.
        assert (await api.get_tasks())["tasks"] == []
    asyncio.run(run())


@pytest.mark.parametrize("status", ["completed", "failed", "completed_empty"])
def test_web_mining_reconnect_replays_existing_tasks_without_losing_downloads(monkeypatch, status):
    import api.main as api

    initial = {"id": "mining-A", "status": "running", "progress": 25, "start_time": 1,
        "logs": ["before reconnect"], "miner": "GP", "config": "test.json"}
    other = {**initial, "id": "mining-B", "start_time": 2}
    download = {"type": "download_progress", "task_id": "download-C", "status": "completed", "progress": 100}
    monkeypatch.setattr(api.TaskManager, "tasks", {initial["id"]: copy.deepcopy(initial), other["id"]: other})
    monkeypatch.setattr(api.TaskManager, "download_tasks", {"download-C": download})
    monkeypatch.setattr(api.manager, "active_connections", [])

    async def run():
        online = ReplaySocket()
        await api.manager.connect(online)
        await api.manager.broadcast({"type": "task_update", "task": initial})
        assert online.messages == [{"type": "task_update", "task": initial}]
        api.manager.disconnect(online)

        terminal = {**initial, "status": status, "progress": 100, "logs": ["before reconnect", "finished offline"],
            "factors": [{"metrics": {"fitness_score": 0.0, "normalized_turnover": float("inf")}}]}
        api.TaskManager.tasks[initial["id"]] = terminal
        await api.manager.broadcast({"type": "task_update", "task": terminal})
        assert (await api.get_tasks())["tasks"][1]["status"] == status

        for _ in range(2):
            socket = ReplaySocket()
            await api.websocket_endpoint(socket)
            matching = [message["task"] for message in socket.messages if message.get("type") == "task_update"]
            assert {task["id"] for task in matching} == {"mining-A", "mining-B"}
            recovered = next(task for task in matching if task["id"] == "mining-A")
            assert recovered["status"] == status
            assert recovered["progress"] == 100
            assert recovered["logs"] == ["before reconnect", "finished offline"]
            assert recovered["factors"][0]["metrics"] == {"fitness_score": 0.0, "normalized_turnover": None}
            assert download in socket.messages
            json.dumps(socket.messages, allow_nan=False)
            assert api.manager.active_connections == []
        assert terminal["factors"][0]["metrics"]["normalized_turnover"] == float("inf")
    asyncio.run(run())


def test_download_diagnostics_are_safe_before_live_broadcast_and_replay(monkeypatch):
    import api.main as api

    secret = "synthetic-private-download-value"
    monkeypatch.setenv("FM_TEST_API_KEY", secret)
    monkeypatch.setattr(api.TaskManager, "download_tasks", {})
    events = []
    def download(self, **kwargs):
        kwargs["progress_callback"](25, "Progress fixture " + secret)
        kwargs["progress_callback"](40, "Unresolved gap: opaque-exception-fixture")
        return {"success": False, "error": "opaque-returned-error-fixture " + secret}
    async def broadcast(event):
        events.append(copy.deepcopy(event))
    monkeypatch.setattr("core.data_feed.batch_downloader.SmartBatchDownloader.download_ohlcv_batch", download)
    monkeypatch.setattr(api.manager, "broadcast", broadcast)
    asyncio.run(api.run_batch_download_task_background("safe-download", "okx", PAIRS[:1], ["1m"],
        "2024-01-01", "2024-01-02", ["futures"], "merge"))
    serialized = json.dumps({"events": events, "replay": api.TaskManager.download_tasks})
    for protected in (secret, "opaque-exception-fixture", "opaque-returned-error-fixture"):
        assert protected not in serialized
    assert events[-1]["status"] == "failed"
    assert events[-1]["results"][0]["error"] == "Download target failed. See local server logs."


def test_websocket_transport_sanitizes_nested_credentials_without_mutating_state():
    from api.ws_manager import ConnectionManager, public_payload

    original = {"type": "task_update", "task": {
        "traceback": "synthetic stack", "details": {"api_key": "synthetic-key"},
        "logs": ["Authorization: Bearer synthetic-token"],
        "metrics": {"IC": .2, "coverage": float("inf")},
    }}
    manager = ConnectionManager()
    socket = ReplaySocket()
    async def run():
        await manager.connect(socket)
        await manager.broadcast(original)
    asyncio.run(run())
    serialized = json.dumps(socket.messages, allow_nan=False)
    assert "synthetic-key" not in serialized and "synthetic-token" not in serialized
    assert "traceback" not in socket.messages[0]["task"]
    assert socket.messages[0]["task"]["metrics"] == {"IC": .2, "coverage": None}
    assert original["task"]["details"]["api_key"] == "synthetic-key"
    assert original["task"]["metrics"]["coverage"] == float("inf")
    assert public_payload(RuntimeError("opaque-error")) == "Operation failed; details are available in local server logs."
