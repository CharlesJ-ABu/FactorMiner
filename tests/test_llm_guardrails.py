import json
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest import mock

import pandas as pd

from core.evaluation.evaluator import (
    FactorOutputError,
    ParallelEvaluator,
    RestrictedSandbox,
    SandboxTimeoutError,
    SecurityError,
)
from core.execution.compiler import FactorCompiler
from core.miner.entities import FactorMetadata, MinerState
from core.miner.expressions import FactorExpressionCode
from core.miner.expressions import FactorExpression
from core.inspector.resolver import FactorResolver
from core.storage.factor_storage import LocalFactorStorage
from user_workspace.custom_miners.my_custom_llm import MyCustomLLMMiner


class StaticDataClient:
    def __init__(self, frame, returns):
        self.frame = frame
        self.returns = returns

    def get_data(self):
        return self.frame

    def get_returns(self):
        return self.returns


class ConcurrencyProbeExpression(FactorExpression):
    def __init__(self, tracker):
        super().__init__()
        self.tracker = tracker

    def compute(self, data):
        with self.tracker["lock"]:
            self.tracker["active"] += 1
            self.tracker["peak"] = max(
                self.tracker["peak"],
                self.tracker["active"],
            )
        time.sleep(0.03)
        with self.tracker["lock"]:
            self.tracker["active"] -= 1
        return data["close"]

    def get_source(self):
        return "concurrency-probe"

    def get_complexity(self):
        return "LoC: 0"

    def to_display_string(self, max_length=None):
        return "concurrency-probe"


class LLMGuardrailTests(unittest.TestCase):
    def setUp(self):
        self.sandbox = RestrictedSandbox()
        self.frame = pd.DataFrame(
            {
                "open": [99.0, 100.0, 102.0],
                "close": [100.0, 101.0, 103.0],
            },
            index=pd.date_range("2026-01-01", periods=3, freq="min"),
        )

    def test_legitimate_open_column_and_factor_are_allowed(self):
        result = self.sandbox.execute_factor_code(
            "factor = df['close'] - df['open']",
            self.frame,
        )
        pd.testing.assert_series_equal(
            result,
            self.frame["close"] - self.frame["open"],
        )

    def test_forbidden_syntax_is_rejected_before_execution(self):
        unsafe = [
            "import os\nfactor = df['close']",
            "factor = open('/tmp/secret')",
            "factor = df.__class__",
            "while True:\n    pass\nfactor = df['close']",
        ]
        for code in unsafe:
            with self.subTest(code=code), self.assertRaises(SecurityError):
                self.sandbox.execute_factor_code(code, self.frame)

    def test_missing_factor_assignment_is_rejected(self):
        with self.assertRaisesRegex(FactorOutputError, "must assign.*'factor'"):
            self.sandbox.execute_factor_code(
                "returns = df['close'].pct_change()",
                self.frame,
            )

    def test_output_contract_rejects_wrong_type_shape_index_and_values(self):
        invalid = [
            "factor = 1",
            "factor = pd.Series([1.0, 2.0])",
            "factor = pd.Series(['a', 'b', 'c'], index=df.index)",
            "factor = df['close'] / 0",
        ]
        for code in invalid:
            with self.subTest(code=code), self.assertRaises(FactorOutputError):
                self.sandbox.execute_factor_code(code, self.frame)

    def test_code_expression_fails_closed_without_sandbox(self):
        expression = FactorExpressionCode("factor = df['close']")
        with self.assertRaisesRegex(RuntimeError, "requires an explicit sandbox"):
            expression.compute(self.frame)

    def test_wall_clock_timeout_terminates_the_worker(self):
        sandbox = RestrictedSandbox(timeout_seconds=0.000001)
        with self.assertRaises(SandboxTimeoutError):
            sandbox.execute_factor_code("factor = df['close']", self.frame)

    def test_evaluation_results_remain_bound_to_their_candidates(self):
        returns = pd.Series([0.01, 0.02, 0.03], index=self.frame.index)
        bad = FactorExpressionCode("factor = 1", sandbox=self.sandbox)
        good = FactorExpressionCode("factor = df['close']", sandbox=self.sandbox)
        evaluator = ParallelEvaluator(StaticDataClient(self.frame, returns), {})

        feedback = evaluator.evaluate([bad, good])

        bad_result = feedback.for_candidate(bad)
        good_result = feedback.for_candidate(good)
        self.assertFalse(bad_result.succeeded)
        self.assertTrue(good_result.succeeded)
        self.assertEqual(good.metrics, good_result.metrics)
        self.assertEqual(feedback.metrics, [good_result.metrics])
        self.assertIs(feedback.execution_status[0]["expr"], bad)

    def test_evaluator_rejects_silent_index_realignment(self):
        returns = pd.Series(
            [0.01, 0.02, 0.03],
            index=self.frame.index.shift(1, freq="D"),
        )
        expression = FactorExpressionCode(
            "factor = df['close']",
            sandbox=self.sandbox,
        )
        feedback = ParallelEvaluator(
            StaticDataClient(self.frame, returns),
            {},
        ).evaluate([expression])
        result = feedback.for_candidate(expression)
        self.assertFalse(result.succeeded)
        self.assertEqual(result.error_type, "FactorOutputError")

    def test_evaluator_respects_configured_worker_limit(self):
        returns = pd.Series([0.01, 0.02, 0.03], index=self.frame.index)
        tracker = {
            "lock": threading.Lock(),
            "active": 0,
            "peak": 0,
        }
        candidates = [ConcurrencyProbeExpression(tracker) for _ in range(4)]
        evaluator = ParallelEvaluator(
            StaticDataClient(self.frame, returns),
            {"evaluation": {"max_workers": 2}},
        )
        feedback = evaluator.evaluate(candidates)
        self.assertEqual(tracker["peak"], 2)
        self.assertTrue(all(result.succeeded for result in feedback.results))

    def test_evaluator_rejects_invalid_worker_limit(self):
        returns = pd.Series([0.01, 0.02, 0.03], index=self.frame.index)
        with self.assertRaisesRegex(ValueError, "between 1 and 64"):
            ParallelEvaluator(
                StaticDataClient(self.frame, returns),
                {"evaluation": {"max_workers": 0}},
            )
        with self.assertRaisesRegex(ValueError, "between 1 and 64"):
            ParallelEvaluator(
                StaticDataClient(self.frame, returns),
                {"evaluation": {"max_workers": 65}},
            )

    def test_cross_asset_llm_output_preserves_both_axes(self):
        columns = pd.Index(["BTC", "ETH"], name="asset")
        cross_data = {
            "open": pd.DataFrame(
                [[10.0, 20.0], [11.0, 19.0], [12.0, 21.0]],
                index=self.frame.index,
                columns=columns,
            ),
            "close": pd.DataFrame(
                [[11.0, 19.0], [12.0, 20.0], [11.0, 22.0]],
                index=self.frame.index,
                columns=columns,
            ),
        }
        returns = pd.DataFrame(
            [[0.01, -0.01], [0.02, 0.01], [-0.01, 0.03]],
            index=self.frame.index,
            columns=columns,
        )
        expression = FactorExpressionCode(
            "factor = df['close'] - df['open']",
            sandbox=self.sandbox,
        )

        feedback = ParallelEvaluator(
            StaticDataClient(cross_data, returns),
            {"evaluation": {"max_workers": 1}},
        ).evaluate([expression])

        result = feedback.for_candidate(expression)
        self.assertTrue(result.succeeded, result.error_message)
        expected = cross_data["close"] - cross_data["open"]
        computed = expression.compute(cross_data)
        pd.testing.assert_frame_equal(computed, expected)

    def test_cross_asset_llm_rejects_series_output(self):
        cross_data = {
            "close": pd.DataFrame(
                [[1.0, 2.0], [2.0, 3.0]],
                columns=["BTC", "ETH"],
            )
        }
        with self.assertRaisesRegex(
            FactorOutputError,
            "must return pandas.DataFrame",
        ):
            self.sandbox.execute_factor_code(
                "factor = df['close'].mean()",
                cross_data,
            )

    def test_reflection_context_includes_successes_and_failures(self):
        state = MinerState()
        state.successful_reflections.append(
            {"code": "factor = df['close']", "metrics": {"IC": 0.1}}
        )
        state.failed_reflections.append(
            {
                "code": "factor = 1",
                "error_type": "FactorOutputError",
                "error": "must return pandas.Series",
            }
        )
        prompt = state.get_llm_context_prompt()
        self.assertIn("past successful", prompt)
        self.assertIn("recent failed", prompt)
        self.assertIn("FactorOutputError", prompt)

    def test_storage_persists_safe_candidate_provenance(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalFactorStorage(temp_dir)
            metadata = FactorMetadata("llm_test", "LLM", "tester")
            provenance = {
                "model": "example-model",
                "prompt_sha256": "abc123",
                "iteration": 2,
                "api_key": "must-not-reach-disk",
                "nested": {"token": "also-secret", "safe": True},
            }
            storage.save_llm_factor(
                "factor = df['close']",
                "reflection",
                metadata,
                provenance=provenance,
            )
            restored = storage.get_metadata("llm_test")
            stored_provenance = restored.logic_reference["provenance"]
            self.assertEqual(stored_provenance["model"], "example-model")
            self.assertTrue(stored_provenance["nested"]["safe"])
            serialized = json.dumps(restored.logic_reference)
            self.assertNotIn("api_key", serialized.lower())
            self.assertNotIn("also-secret", serialized)
            self.assertNotIn("must-not-reach-disk", serialized)

    def test_custom_llm_candidates_always_receive_the_sandbox(self):
        miner = MyCustomLLMMiner(
            self.frame,
            {
                "population_size": 2,
                "data_feeds": {"required_streams": ["open", "close"]},
            },
        )
        miner.initialize_search_space()
        candidates = miner.generate_candidates()
        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(candidate.sandbox is miner.sandbox for candidate in candidates))

    def test_post12_live_experiment_fails_closed_without_credentials(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            miner = MyCustomLLMMiner(
                self.frame,
                {
                    "population_size": 1,
                    "data_feeds": {"required_streams": ["open", "close"]},
                    "llm_api_config": {
                        "keys_env": ["POST12_TEST_API_KEY"],
                        "model": "test-model",
                    },
                    "experiment": {
                        "record_dir": temp_dir,
                        "require_live_api": True,
                        "allow_fallback": False,
                    },
                },
            )
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("POST12_TEST_API_KEY", None)
                with self.assertRaisesRegex(RuntimeError, "requires a live LLM API"):
                    miner.initialize_search_space()
            events = [
                json.loads(line)
                for line in (
                    Path(temp_dir) / "events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(events[-1]["event_type"], "run_blocked")

    def test_custom_llm_resolves_openai_compatible_environment_config(self):
        miner = MyCustomLLMMiner(
            self.frame,
            {
                "llm_api_config": {
                    "keys_env": ["TEST_AI_API_KEY"],
                    "model_env": "TEST_AI_MODEL",
                    "base_url_env": "TEST_AI_API_BASE",
                    "model": "default-model",
                    "base_url": "https://example.invalid",
                }
            },
        )
        with mock.patch.dict(
            os.environ,
            {
                "TEST_AI_API_KEY": "test-secret-value",
                "TEST_AI_MODEL": "test-model",
                "TEST_AI_API_BASE": "https://api.example.test/v1",
            },
            clear=False,
        ):
            resolved = miner._resolved_api_config()
        self.assertEqual(resolved["keys"], ["test-secret-value"])
        self.assertEqual(resolved["model"], "test-model")
        self.assertEqual(
            resolved["base_url"],
            "https://api.example.test/v1/chat/completions",
        )

    def test_experiment_recorder_keeps_generation_evaluation_and_reflection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            miner = MyCustomLLMMiner(
                StaticDataClient(
                    self.frame,
                    pd.Series([0.01, 0.02, 0.03], index=self.frame.index),
                ),
                {
                    "population_size": 1,
                    "data_feeds": {"required_streams": ["open", "close"]},
                    "experiment": {
                        "record_dir": temp_dir,
                        "require_live_api": False,
                        "allow_fallback": True,
                    },
                    "evaluation": {"max_workers": 1},
                },
            )
            miner.evaluator = ParallelEvaluator(miner.data, miner.config)
            with mock.patch.object(
                miner,
                "_get_fallback_code",
                return_value="factor = df['close'] - df['open']",
            ):
                miner.mine(1)
            events = [
                json.loads(line)
                for line in (
                    Path(temp_dir) / "events.jsonl"
                ).read_text(encoding="utf-8").splitlines()
            ]
            event_types = {event["event_type"] for event in events}
            self.assertIn("candidate_generated", event_types)
            self.assertIn("candidate_evaluated", event_types)
            self.assertIn("reflection_updated", event_types)
            evaluation = next(
                event
                for event in events
                if event["event_type"] == "candidate_evaluated"
            )
            self.assertTrue(evaluation["diagnostics"]["deterministic_replay_equal"])

    def test_inspector_restores_llm_source_with_a_sandbox(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalFactorStorage(temp_dir)
            metadata = FactorMetadata("inspect_llm", "LLM", "tester")
            storage.save_llm_factor(
                "factor = df['close']",
                "prior reflection",
                metadata,
                provenance={"prompt_sha256": "abc"},
            )
            expression = FactorResolver(storage=storage).resolve(
                factor_id="inspect_llm"
            )
            result = expression.compute(self.frame)
            pd.testing.assert_series_equal(result, self.frame["close"])
            self.assertEqual(expression.get_reflection_history(), "prior reflection")

    def test_storage_source_contract_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalFactorStorage(temp_dir)
            with self.assertRaisesRegex(ValueError, "must not contain a path"):
                storage.load_llm_source("../outside.py")

    def test_compiler_reconstructs_llm_and_gp_as_callable_factors(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalFactorStorage(temp_dir)
            llm_metadata = FactorMetadata("compiled_llm", "LLM", "tester")
            storage.save_llm_factor(
                "factor = df['close'] - df['open']",
                "",
                llm_metadata,
            )
            gp_metadata = FactorMetadata("compiled_gp", "MyCustomGP", "tester")
            storage.save_gp_factor(
                {"op": "sub", "left": "close", "right": "open"},
                gp_metadata,
            )
            compiler = FactorCompiler(storage)

            llm_values = compiler.compile_for_live_trading("compiled_llm")(
                self.frame
            )
            gp_values = compiler.compile_for_live_trading("compiled_gp")(
                self.frame
            )

            expected = self.frame["close"] - self.frame["open"]
            pd.testing.assert_series_equal(llm_values, expected)
            pd.testing.assert_series_equal(gp_values, expected)

    def test_deployment_never_reports_success_without_a_transport(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = LocalFactorStorage(temp_dir)
            metadata = FactorMetadata("deploy_llm", "LLM", "tester")
            storage.save_llm_factor("factor = df['close']", "", metadata)
            with self.assertRaisesRegex(NotImplementedError, "was not deployed"):
                FactorCompiler(storage).deploy_to_live_server(
                    "deploy_llm",
                    "grpc://example.invalid",
                )


if __name__ == "__main__":
    unittest.main()
