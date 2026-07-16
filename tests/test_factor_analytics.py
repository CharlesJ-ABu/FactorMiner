import tempfile
import unittest
import os

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

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


if __name__ == "__main__":
    unittest.main()
