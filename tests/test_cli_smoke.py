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
        if miner == "MyCustomNN":
            config.update({"population_size": 1, "hidden_dim": 2, "learning_rate": 0.01})
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

    def test_gp_cli_smoke(self):
        self._run_miner("MyCustomGP")

    def test_rl_cli_smoke(self):
        self._run_miner("MyCustomRL")

    def test_llm_cli_smoke(self):
        self._run_miner("MyCustomLLM")

    def test_nn_cli_smoke(self):
        self._run_miner("MyCustomNN")


if __name__ == "__main__":
    unittest.main()
