import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.data_feed.naming import data_path
from core.data_feed.real_client import RealDataClient
from core.evaluation.targets import (
    ForwardReturnTarget,
    TargetConfigError,
)
from core.inspector.engine import FactorInspectorEngine
from core.miner.entities import FactorMetadata
from core.startup_validation import StartupValidationError, validate_mining_startup
from core.storage.factor_storage import LocalFactorStorage


PAIR = "BTC/USDT:USDT"


def fixture_frame(periods: int = 32) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=periods, freq="min")
    close = pd.Series(100.0 + np.arange(periods), dtype=float)
    return pd.DataFrame(
        {
            "date": dates,
            "open": close - 0.25,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000.0 + np.arange(periods),
        }
    )


class ForwardReturnTargetTests(unittest.TestCase):
    def test_default_target_preserves_legacy_one_bar_close_return(self):
        frame = fixture_frame().set_index("date")
        target = ForwardReturnTarget.from_config(None)
        expected = frame["close"].shift(-1) / frame["close"] - 1.0

        pd.testing.assert_series_equal(target.build(frame), expected.rename("returns"))
        self.assertEqual(target.definition(), "close[t+1] / close[t] - 1")

    def test_next_open_three_bar_target_has_exact_time_alignment(self):
        frame = fixture_frame().set_index("date")
        target = ForwardReturnTarget.from_config(
            {
                "entry_price": "next_open",
                "exit_price": "close",
                "horizon_bars": 3,
            }
        )
        actual = target.build(frame)
        expected = frame["close"].shift(-3) / frame["open"].shift(-1) - 1.0

        pd.testing.assert_series_equal(actual, expected.rename("returns"))
        self.assertEqual(actual.tail(3).isna().sum(), 3)
        self.assertEqual(
            target.definition(),
            "close[t+3] / open[t+1] - 1",
        )

    def test_invalid_target_fields_fail_fast(self):
        with self.assertRaises(TargetConfigError):
            ForwardReturnTarget.from_config({"horizon_bars": 0})
        with self.assertRaises(TargetConfigError):
            ForwardReturnTarget.from_config({"entry_price": "same_close"})
        with self.assertRaises(TargetConfigError):
            ForwardReturnTarget.from_config({"horizonn_bars": 3})

    def test_startup_validation_rejects_invalid_target(self):
        config = {
            "paradigm": "GP",
            "max_iterations": 1,
            "target": {"horizon_bars": True},
            "data_feeds": {
                "pairs": [PAIR],
                "required_streams": ["close", "volume"],
            },
        }
        with self.assertRaisesRegex(
            StartupValidationError,
            "target.horizon_bars must be a positive integer",
        ):
            validate_mining_startup(config)


class TargetIntegrationTests(unittest.TestCase):
    def _write_data(self, root: Path, periods: int = 32) -> None:
        target = data_path(root / "data", "binance", PAIR, "1m", "futures")
        target.parent.mkdir(parents=True, exist_ok=True)
        fixture_frame(periods).to_feather(target)

    def test_disjoint_periods_do_not_create_cross_boundary_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_data(root)
            config = {
                "target": {"horizon_bars": 1},
                "data_feeds": {
                    "exchange": "binance",
                    "instrument_type": "futures",
                    "timeframe": "1m",
                    "pairs": [PAIR],
                    "mine_period": [
                        ["2024-01-01 00:00:00", "2024-01-01 00:02:00"],
                        ["2024-01-01 00:07:00", "2024-01-01 00:09:00"],
                    ],
                    "test_period": [],
                    "mining_mode": "sequential_single",
                },
            }
            previous_cwd = os.getcwd()
            os.chdir(root)
            try:
                returns = RealDataClient(config).get_returns()
            finally:
                os.chdir(previous_cwd)

            self.assertTrue(pd.isna(returns.loc["2024-01-01 00:02:00"]))
            self.assertTrue(pd.isna(returns.loc["2024-01-01 00:09:00"]))
            self.assertAlmostEqual(
                returns.loc["2024-01-01 00:07:00"],
                108.0 / 107.0 - 1.0,
            )

    def test_inspector_reuses_target_stored_in_factor_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_data(root, periods=64)
            previous_cwd = os.getcwd()
            os.chdir(root)
            try:
                target = ForwardReturnTarget.from_config(
                    {
                        "entry_price": "next_open",
                        "exit_price": "close",
                        "horizon_bars": 3,
                    }
                )
                metadata = FactorMetadata(
                    factor_id="fac_target",
                    miner_type="GP",
                    user_id="test",
                    data_lineage={
                        "target": target.as_dict(),
                        "forward_return_definition": target.definition(),
                    },
                )
                LocalFactorStorage("factor_db").save_gp_factor("close", metadata)
                results = FactorInspectorEngine().inspect(
                    factor_id="fac_target",
                    pairs=[PAIR],
                    periods=[["2024-01-01", "2024-01-01 00:59:00"]],
                    timeframe="1m",
                )
            finally:
                os.chdir(previous_cwd)

            metrics = results[PAIR]
            self.assertEqual(metrics["target"], target.as_dict())
            self.assertEqual(
                metrics["forward_return_definition"],
                "close[t+3] / open[t+1] - 1",
            )
            frame = fixture_frame(64).set_index("date").iloc[:60]
            expected_returns = target.build(frame)
            expected_ic = frame["close"].corr(expected_returns, method="pearson")
            self.assertAlmostEqual(metrics["pearson"]["overall"], expected_ic)
            expected_lag_one = frame["close"].corr(
                target.with_horizon(1).build(frame),
                method="spearman",
            )
            self.assertAlmostEqual(metrics["decay"]["Lag_1"], expected_lag_one)


if __name__ == "__main__":
    unittest.main()
