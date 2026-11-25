import tempfile
import unittest
import os
import asyncio
import json
import math

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocket

from core.analysis.factor_analytics import analyze_factor_snapshot
from core.miner.entities import FactorMetadata
from core.storage.factor_storage import LocalFactorStorage


class FactorAnalyticsTests(unittest.TestCase):
    def test_sequential_snapshot_persists_lineage_and_real_tearsheet_metrics(self):
        with tempfile.TemporaryDirectory() as root:
            storage = LocalFactorStorage(root)
            metadata = FactorMetadata(factor_id="fac_test", miner_type="MyCustomGP", user_id="test")
            storage.save_gp_factor({"op": "add", "left": "close", "right": "volume"}, metadata)

            index = pd.date_range("2024-01-01", periods=48, freq="h")
            factor = pd.Series(np.linspace(-1, 1, len(index)), index=index)
            returns = pd.Series(np.linspace(-0.01, 0.01, len(index)), index=index)
            saved = storage.save_factor_snapshot(
                "fac_test",
                factor,
                returns,
                {"pairs": ["BTC/USDT:USDT"], "timeframe": "1h"},
            )

            self.assertTrue(saved)
            snapshot = storage.load_factor_values("fac_test")
            self.assertEqual(set(snapshot.columns), {"timestamp", "factor", "forward_return"})
            refreshed = storage.get_metadata("fac_test")
            self.assertEqual(refreshed.data_lineage["snapshot_rows"], 48)
            self.assertEqual(refreshed.data_lineage["pairs"], ["BTC/USDT:USDT"])

            analysis = analyze_factor_snapshot(snapshot, rolling_window=12)
            self.assertEqual(analysis["mode"], "sequential_single")
            self.assertEqual(analysis["summary"]["observations"], 48)
            self.assertTrue(analysis["rolling_ic"])
            self.assertTrue(analysis["quantiles"])
            self.assertTrue(analysis["turnover"])

    def test_cross_asset_snapshot_uses_cross_sectional_ic(self):
        with tempfile.TemporaryDirectory() as root:
            storage = LocalFactorStorage(root)
            metadata = FactorMetadata(factor_id="fac_cross", miner_type="MyCustomGP", user_id="test")
            storage.save_gp_factor({"op": "sub", "left": "close", "right": "volume"}, metadata)

            index = pd.date_range("2024-01-01", periods=16, freq="h")
            columns = ["BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT"]
            factor = pd.DataFrame(
                [np.arange(len(columns)) + offset for offset in range(len(index))],
                index=index,
                columns=columns,
            )
            returns = factor * 0.001
            self.assertTrue(storage.save_factor_snapshot("fac_cross", factor, returns, {"mining_mode": "cross_asset"}))

            analysis = analyze_factor_snapshot(storage.load_factor_values("fac_cross"), rolling_window=5)
            self.assertEqual(analysis["mode"], "cross_asset")
            self.assertTrue(analysis["rolling_ic"])
            self.assertTrue(analysis["quantiles"])

    def test_inspector_analysis_compare_and_batch_review_endpoints(self):
        from api.main import app

        with tempfile.TemporaryDirectory() as root:
            previous_cwd = os.getcwd()
            os.chdir(root)
            try:
                storage = LocalFactorStorage("factor_db")
                index = pd.date_range("2024-01-01", periods=32, freq="h")
                returns = pd.Series(np.linspace(-0.01, 0.01, len(index)), index=index)
                for factor_id, multiplier in (("fac_api_one", 1.0), ("fac_api_two", -1.0)):
                    metadata = FactorMetadata(factor_id=factor_id, miner_type="MyCustomGP", user_id="test")
                    storage.save_gp_factor({"op": "add", "left": "close", "right": "volume"}, metadata)
                    storage.save_factor_snapshot(
                        factor_id,
                        pd.Series(np.linspace(-1, 1, len(index)) * multiplier, index=index),
                        returns,
                        {"pairs": ["BTC/USDT:USDT"]},
                    )

                client = TestClient(app)
                analysis = client.get("/api/factors/fac_api_one/analysis")
                self.assertEqual(analysis.status_code, 200, analysis.text)
                self.assertEqual(analysis.json()["analysis"]["summary"]["observations"], 32)

                comparison = client.post("/api/factors/compare", json={"factor_ids": ["fac_api_one", "fac_api_two"]})
                self.assertEqual(comparison.status_code, 200, comparison.text)
                self.assertEqual(len(comparison.json()["factors"]), 2)

                batch = client.patch(
                    "/api/factors/lifecycle/batch",
                    json={"factor_ids": ["fac_api_one", "fac_api_two"], "lifecycle_status": "INSPECTED"},
                )
                self.assertEqual(batch.status_code, 200, batch.text)
                self.assertEqual(len(batch.json()["updated"]), 2)
            finally:
                os.chdir(previous_cwd)


@pytest.mark.parametrize("diagnostic", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_diagnostics_do_not_break_http_results_or_change_stored_metrics(
    tmp_path, monkeypatch, diagnostic,
):
    import api.main as api

    monkeypatch.chdir(tmp_path)
    storage = LocalFactorStorage("factor_db")
    metrics = {"IC": 0.0, "RankIC": 0.0, "Turnover": 0.0, "fitness_score": 0.0,
               "diagnostic": diagnostic, "nested": {"values": [diagnostic, -3.0]}}
    storage.save_gp_factor(0, FactorMetadata("delivery", "GP", "test", metrics=metrics))
    path = tmp_path / "factor_db" / "metadata" / "delivery.json"
    original = path.read_bytes()
    task = {"id": "delivery-task", "start_time": "2024-01-01", "status": "completed",
            "factors": [{"metrics": metrics}]}
    monkeypatch.setattr(api.TaskManager, "tasks", {task["id"]: task})
    monkeypatch.setattr("core.storage.factor_storage.get_global_storage", lambda: storage)
    with TestClient(api.app, raise_server_exceptions=False) as client:
        responses = {route: client.get(route) for route in (
            "/api/factors", "/api/factors/delivery", "/api/dashboard", "/api/tasks", "/api/stats",
        )}
    for route, response in responses.items():
        assert response.status_code == 200, (route, response.text)
        json.loads(response.text, parse_constant=lambda value: pytest.fail(f"Non-JSON token: {value}"))
    received_metrics = [
        responses["/api/factors"].json()["factors"][0]["metrics"],
        responses["/api/factors/delivery"].json()["metadata"]["metrics"],
        responses["/api/dashboard"].json()["factors"]["top_by_fitness"][0]["metrics"],
        responses["/api/tasks"].json()["tasks"][0]["factors"][0]["metrics"],
        responses["/api/stats"].json()["recent_activity"][0]["factors"][0]["metrics"],
    ]
    for actual in received_metrics:
        assert actual["fitness_score"] == 0.0
        assert actual["diagnostic"] is None
        assert actual["nested"]["values"] == [None, -3.0]
    assert path.read_bytes() == original
    assert not math.isfinite(metrics["diagnostic"])
    assert not math.isfinite(task["factors"][0]["metrics"]["diagnostic"])


@pytest.mark.parametrize("diagnostic", [float("inf"), float("-inf"), float("nan")])
def test_websocket_metrics_use_strict_json_without_mutating_task(diagnostic):
    from api.ws_manager import ConnectionManager

    event = {"type": "task_update", "task": {"id": "test", "status": "completed",
        "factors": [{"metrics": {"fitness_score": -3.0, "diagnostic": [diagnostic, 0.0]}}]}}

    async def run():
        sent = []
        async def receive():
            return {"type": "websocket.connect"}
        async def send(message):
            sent.append(message)
        socket = WebSocket({"type": "websocket", "path": "/ws/monitor", "headers": [],
                            "query_string": b""}, receive, send)
        manager = ConnectionManager()
        await manager.connect(socket)
        await manager.broadcast(event)
        wire = next(message["text"] for message in sent if message["type"] == "websocket.send")
        return json.loads(wire, parse_constant=lambda value: pytest.fail(f"Non-JSON token: {value}"))

    actual = asyncio.run(run())
    assert actual["type"] == "task_update" and actual["task"]["status"] == "completed"
    assert actual["task"]["factors"][0]["metrics"] == {"fitness_score": -3.0, "diagnostic": [None, 0.0]}
    assert not math.isfinite(event["task"]["factors"][0]["metrics"]["diagnostic"][0])


if __name__ == "__main__":
    unittest.main()
