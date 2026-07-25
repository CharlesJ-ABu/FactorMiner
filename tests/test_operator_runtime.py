import unittest

import pandas as pd

from core.miner.operator_runtime import OperatorRuntimeError, evaluate_ast, resolve_operator_specs
from core.miner.registry import (
    EvaluatorRegistry,
    ExtensionRegistrationError,
    MinerRegistry,
    OperatorRegistry,
)
from core.startup_validation import StartupValidationError, validate_mining_startup
from core.utils.dynamic_loader import load_user_modules


class OperatorRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = load_user_modules("user_workspace")
        if cls.report.errors:
            raise AssertionError(cls.report.errors)

    def test_registered_unary_operator_evaluates_an_ast(self):
        data = pd.DataFrame({"close": range(1, 20), "volume": range(20, 39)})
        for operator in ("custom_ts_decay", "ts_zscore_20", "ts_delta_5", "ts_rank_20", "ts_volatility_20"):
            result = evaluate_ast({"op": operator, "left": "close"}, data)
            self.assertIsInstance(result, pd.Series)
            self.assertEqual(result.index.tolist(), data.index.tolist())
            self.assertGreater(result.notna().sum(), 0, msg=operator)

    def test_operator_metadata_exposes_unary_arity(self):
        specs = resolve_operator_specs(["custom_ts_decay", "ts_zscore_20", "ts_delta_5", "ts_rank_20", "ts_volatility_20", "add"])
        self.assertEqual(specs["custom_ts_decay"]["arity"], 1)
        self.assertEqual(specs["ts_zscore_20"]["arity"], 1)
        self.assertEqual(specs["ts_delta_5"]["arity"], 1)
        self.assertEqual(specs["ts_rank_20"]["arity"], 1)
        self.assertEqual(specs["ts_volatility_20"]["arity"], 1)
        self.assertEqual(specs["add"]["arity"], 2)

    def test_unknown_operator_has_actionable_startup_error(self):
        config = {
            "paradigm": "MyCustomGP",
            "max_iterations": 1,
            "data_feeds": {"pairs": ["BTC/USDT:USDT"], "required_streams": ["close"]},
            "search_space": {"allowed_operators": ["not_registered"]},
        }

        with self.assertRaisesRegex(StartupValidationError, "Unknown operator 'not_registered'"):
            validate_mining_startup(config, self.report)

    def test_unknown_fitness_hook_has_actionable_startup_error(self):
        config = {
            "paradigm": "MyCustomGP",
            "max_iterations": 1,
            "data_feeds": {"pairs": ["BTC/USDT:USDT"], "required_streams": ["close"]},
            "fitness": {"hook": "missing_hook"},
        }

        with self.assertRaisesRegex(StartupValidationError, "Unknown Fitness Hook 'missing_hook'"):
            validate_mining_startup(config, self.report)

    def test_unknown_miner_has_actionable_startup_error(self):
        config = {
            "paradigm": "MissingMiner",
            "max_iterations": 1,
            "data_feeds": {"pairs": ["BTC/USDT:USDT"], "required_streams": ["close"]},
        }

        with self.assertRaisesRegex(StartupValidationError, "Unknown Miner 'MissingMiner'"):
            validate_mining_startup(config, self.report)

    def test_deprecated_dl_name_is_mapped_to_nn(self):
        config = {
            "paradigm": "DL",
            "max_iterations": 1,
            "data_feeds": {
                "pairs": ["BTC/USDT:USDT"],
                "required_streams": ["close", "volume"],
            },
        }
        with self.assertLogs("core.startup_validation", level="WARNING") as logs:
            validate_mining_startup(config, self.report)
        self.assertEqual(config["paradigm"], "NN")
        self.assertIn("deprecated", "\n".join(logs.output))
        self.assertIn("NN", MinerRegistry._registry)

    def test_invalid_extension_signatures_fail_at_registration(self):
        with self.assertRaisesRegex(ExtensionRegistrationError, "arity must be 1 or 2"):
            OperatorRegistry.register(arity=3)

        def incomplete_hook(base_metrics):
            return 0.0

        with self.assertRaisesRegex(ExtensionRegistrationError, "must accept at least 3"):
            EvaluatorRegistry.register_fitness_hook("incomplete_hook")(incomplete_hook)

    def test_invalid_operator_name_fails_without_silent_zero_factor(self):
        data = pd.DataFrame({"close": range(1, 5)})
        with self.assertRaisesRegex(OperatorRuntimeError, "Unknown operator"):
            evaluate_ast({"op": "missing", "left": "close"}, data)


if __name__ == "__main__":
    unittest.main()
