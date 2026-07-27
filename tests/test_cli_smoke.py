import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

from core.data_feed.naming import data_path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
USER_WORKSPACE = PROJECT_ROOT / "user_workspace"
PAIR = "BTC/USDT:USDT"


class CLISmokeTests(unittest.TestCase):
    """Run every shipped custom paradigm through the public CLI on fixture data."""

    def _write_fixture_data(self, root: Path) -> None:
        dates = pd.date_range("2024-01-01", periods=180, freq="min")
        close = 100 + np.sin(np.linspace(0, 12, len(dates))) + np.linspace(0, 2, len(dates))
        frame = pd.DataFrame(
            {
                "date": dates,
                "open": close - 0.1,
                "high": close + 0.2,
                "low": close - 0.2,
                "close": close,
                "volume": 1000 + np.linspace(0, 100, len(dates)),
            }
        )
        target = data_path(root / "data", "binance", PAIR, "1m", "futures")
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_feather(target)

    def _config_for(self, miner: str) -> dict:
        config = {
            "population_size": 2,
            "max_iterations": 1,
            "top_k_factors": 2,
            "data_feeds": {
                "required_streams": ["close", "volume"],
                "exchange": "binance",
                "instrument_type": "futures",
                "timeframe": "1m",
                "pairs": [PAIR],
                "mine_period": [["2024-01-01", "2024-01-01 02:30:00"]],
                "test_period": [["2024-01-01", "2024-01-01 02:30:00"]],
                "mining_mode": "sequential_single",
            },
        }
        if miner in {"MyCustomGP", "MyCustomRL"}:
            config["search_space"] = {
                "allowed_operators": [
                    "add", "sub", "mul", "div", "custom_ts_decay",
                    "ts_zscore_20", "ts_delta_5", "ts_rank_20", "ts_volatility_20",
                ]
            }
        if miner == "MyCustomRL":
            config["rl_config"] = {"learning_rate": 0.1, "max_depth": 2}
        if miner == "MyCustomLLM":
            config["population_size"] = 1
        if miner in {"NN", "MyCustomNN", "MyTemporalNN"}:
            config.update({"population_size": 1, "hidden_dim": 2, "learning_rate": 0.01})
            config["nn_training_epochs"] = 5
            config["data_feeds"]["mine_period"] = [
                ["2024-01-01", "2024-01-01 01:39:00"]
            ]
            config["data_feeds"]["test_period"] = [
                ["2024-01-01 01:40:00", "2024-01-01 02:59:00"]
            ]
        if miner == "MyTemporalNN":
            config["data_feeds"]["required_streams"] = [
                "open", "high", "low", "close", "volume"
            ]
            config["target"] = {
                "type": "forward_return",
                "entry_price": "next_open",
                "exit_price": "close",
                "horizon_bars": 5,
                "return_type": "simple",
            }
            config["nn_ic_loss_weight"] = 0.7
            config["nn_min_fitness"] = -100.0
            config["fitness"] = {"hook": "my_temporal_positive_ic"}
        return config

    def _run_miner(self, miner: str) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self._write_fixture_data(root)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(self._config_for(miner)), encoding="utf-8")
            environment = os.environ.copy()
            python_path = environment.get("PYTHONPATH", "")
            environment["PYTHONPATH"] = f"{PROJECT_ROOT}{os.pathsep}{python_path}" if python_path else str(PROJECT_ROOT)

            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    "-m",
                    "core.cli",
                    "mine",
                    "--miner",
                    miner,
                    "--config",
                    str(config_path),
                    "--user-dir",
                    str(USER_WORKSPACE),
                    "--iterations",
                    "1",
                ],
                cwd=root,
                env=environment,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, msg=result.stdout + "\n" + result.stderr)
            self.assertIn("FactorMiner execution completed successfully.", result.stderr)
            self.assertTrue(
                list((root / "factor_db" / "metadata").glob("*.json")),
                msg=result.stdout + "\n" + result.stderr,
            )
            self.assertTrue(
                list((root / "factor_db" / "values").glob("*.parquet")),
                msg=result.stdout + "\n" + result.stderr,
            )
            if miner in {"NN", "MyCustomNN", "MyTemporalNN"}:
                metadata_files = list((root / "factor_db" / "metadata").glob("*.json"))
                metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
                logic = metadata["logic_reference"]
                self.assertEqual(logic["type"], "nn_channel")
                expected_format = (
                    "numpy_temporal_ic_mlp_v1"
                    if miner == "MyTemporalNN"
                    else "numpy_mlp_factor_v1"
                )
                self.assertEqual(logic["model_format"], expected_format)
                self.assertTrue((root / "factor_db" / "models" / logic["model_file"]).is_file())
                self.assertEqual(metadata["metrics"]["out_of_sample"], 1.0)
                self.assertEqual(metadata["data_lineage"]["evaluation_split"], "test")
                if miner == "MyTemporalNN":
                    expected_target = {
                        "type": "forward_return",
                        "entry_price": "next_open",
                        "exit_price": "close",
                        "horizon_bars": 5,
                        "return_type": "simple",
                    }
                    self.assertEqual(
                        metadata["data_lineage"]["target"],
                        expected_target,
                    )
                    self.assertEqual(
                        metadata["data_lineage"]["forward_return_definition"],
                        "close[t+5] / open[t+1] - 1",
                    )
                    snapshot_file = root / "factor_db" / "values" / (
                        metadata["data_lineage"]["snapshot_file"]
                    )
                    snapshot = pd.read_parquet(snapshot_file)
                    prices = pd.read_feather(
                        data_path(
                            root / "data",
                            "binance",
                            PAIR,
                            "1m",
                            "futures",
                        )
                    ).set_index("date")
                    first = snapshot.iloc[0]
                    location = prices.index.get_loc(pd.Timestamp(first["timestamp"]))
                    expected_return = (
                        prices["close"].iloc[location + 5]
                        / prices["open"].iloc[location + 1]
                        - 1.0
                    )
                    self.assertAlmostEqual(
                        first["forward_return"],
                        expected_return,
                    )

    def test_gp_cli_smoke(self):
        self._run_miner("MyCustomGP")

    def test_rl_cli_smoke(self):
        self._run_miner("MyCustomRL")

    def test_llm_cli_smoke(self):
        self._run_miner("MyCustomLLM")

    def test_nn_cli_smoke(self):
        self._run_miner("NN")

    def test_temporal_nn_cli_smoke(self):
        self._run_miner("MyTemporalNN")


if __name__ == "__main__":
    unittest.main()
