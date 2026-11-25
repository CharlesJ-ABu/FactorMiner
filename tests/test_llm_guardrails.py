import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import tempfile
import threading
import time
import unittest
from unittest import mock

import pandas as pd
import pytest

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
from core.miner.expressions import FactorExpression, FactorExpressionAST
from core.miner.registry import EvaluatorRegistry
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

    def test_concurrent_workers_limit_threads_without_changing_parent_environment(self):
        thread_names = (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "OMP_THREAD_LIMIT",
            "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            "BLIS_NUM_THREADS",
        )
        parent_settings = {name: "8" for name in thread_names}
        parent_settings["FACTOR_SANDBOX_ENV_PROBE"] = "inherited"
        real_popen = subprocess.Popen
        child_environments = []

        with mock.patch.dict(os.environ, parent_settings):
            parent_environment = dict(os.environ)

            def start_worker(*args, **kwargs):
                self.assertEqual(dict(os.environ), parent_environment)
                child_environments.append(kwargs["env"])
                return real_popen(*args, **kwargs)

            frames = [self.frame, self.frame + 10.0]
            with mock.patch(
                "core.evaluation.code_sandbox.subprocess.Popen",
                side_effect=start_worker,
            ), ThreadPoolExecutor(max_workers=2) as executor:
                results = list(executor.map(
                    lambda frame: self.sandbox.execute_factor_code(
                        "factor = df['close']", frame
                    ),
                    frames,
                ))

            self.assertEqual(dict(os.environ), parent_environment)
            self.assertEqual(len(child_environments), 2)
            self.assertIsNot(child_environments[0], child_environments[1])
            for environment in child_environments:
                self.assertEqual(environment["FACTOR_SANDBOX_ENV_PROBE"], "inherited")
                self.assertTrue(all(environment[name] == "1" for name in thread_names))
            for result, frame in zip(results, frames):
                pd.testing.assert_series_equal(result, frame["close"])

    def test_worker_starts_from_temporary_directory_without_pythonpath(self):
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.dict(os.environ):
            os.environ.pop("PYTHONPATH", None)
            try:
                os.chdir(temp_dir)
                result = self.sandbox.execute_factor_code(
                    "factor = df['close']", self.frame
                )
            finally:
                os.chdir(original_cwd)
        pd.testing.assert_series_equal(result, self.frame["close"])

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
        workers = []
        real_popen = subprocess.Popen

        def start_worker(*args, **kwargs):
            worker = real_popen(*args, **kwargs)
            workers.append(worker)
            return worker

        with mock.patch(
            "core.evaluation.code_sandbox.subprocess.Popen",
            side_effect=start_worker,
        ), self.assertRaises(SandboxTimeoutError):
            sandbox.execute_factor_code("factor = df['close']", self.frame)
        self.assertEqual(len(workers), 1)
        self.assertIsNotNone(workers[0].poll())
        self.assertTrue(workers[0].stdin.closed)
        self.assertTrue(workers[0].stdout.closed)
        self.assertTrue(workers[0].stderr.closed)

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


@pytest.mark.parametrize("invalid_score", [
    float("nan"), float("inf"), float("-inf"),
    {"fitness_score": float("nan")}, {"fitness_score": float("inf")},
    {"fitness_score": float("-inf")}, {},
])
def test_fitness_hook_rejects_non_finite_or_missing_score_without_losing_other_candidate(
    monkeypatch, invalid_score,
):
    index = pd.date_range("2024-01-01", periods=40, freq="h")
    frame = pd.DataFrame({"close": range(100, 140), "volume": range(40)}, index=index)
    returns = pd.Series([0.01, -0.02, 0.03, -0.01] * 10, index=index)
    bad, good = FactorExpressionAST("close"), FactorExpressionAST("volume")
    hook_name = "review_finite_fitness"
    monkeypatch.setitem(
        EvaluatorRegistry._registry, hook_name,
        lambda factor, target, metrics: invalid_score if factor.name == "close" else 3.0,
    )
    feedback = ParallelEvaluator(
        StaticDataClient(frame, returns), {"fitness": {"hook": hook_name}},
    ).evaluate([bad, good])
    invalid_result = feedback.for_candidate(bad)
    valid_result = feedback.for_candidate(good)
    assert not invalid_result.succeeded
    assert invalid_result.error_type == "FactorOutputError"
    assert "fitness_score" in invalid_result.error_message
    assert "finite" in invalid_result.error_message
    assert bad.metrics == {}
    assert valid_result.succeeded
    assert valid_result.metrics["fitness_score"] == 3.0
    assert feedback.metrics == [valid_result.metrics]
    assert feedback.execution_status[0]["expr"] is bad


@pytest.mark.parametrize("score", [0.0, -3.5, {"fitness_score": 0.0, "extra": 2.0},
                                   {"fitness_score": -3.5, "extra": 2.0}])
def test_fitness_hook_preserves_finite_zero_negative_scores_and_extra_metrics(monkeypatch, score):
    frame = pd.DataFrame({"close": range(40)})
    returns = pd.Series([0.01, -0.02, 0.03, -0.01] * 10)
    hook_name = "review_valid_fitness"
    monkeypatch.setitem(EvaluatorRegistry._registry, hook_name, lambda *args: score)
    candidate = FactorExpressionAST("close")
    result = ParallelEvaluator(
        StaticDataClient(frame, returns), {"fitness": {"hook": hook_name}},
    ).evaluate([candidate]).for_candidate(candidate)
    assert result.succeeded, result.error_message
    expected = score["fitness_score"] if isinstance(score, dict) else score
    assert result.metrics["fitness_score"] == expected
    assert candidate.metrics == result.metrics
    if isinstance(score, dict):
        assert result.metrics["extra"] == 2.0


def _publication_archive_director(tmp_path, monkeypatch, candidates, **config):
    from core.miner import director as module

    index = pd.date_range("2025-01-01", periods=4, freq="min")
    frame = pd.DataFrame({"close": [10., 11., 12., 13.]}, index=index)
    client = StaticDataClient(frame, pd.Series([.1, .2, -.1, .05], index=index))
    storage = LocalFactorStorage(str(tmp_path / "factor_db"))
    miner = mock.Mock()
    miner.mine.return_value = candidates
    monkeypatch.setattr(module, "get_global_storage", lambda: storage)
    monkeypatch.setattr(module.FactorMinerDirector, "_instantiate_miner", lambda *args: miner)
    director = module.FactorMinerDirector({"paradigm": "GP", **config}, client)
    monkeypatch.setattr(director, "_print_results_table", mock.Mock())
    return director, storage


@pytest.mark.parametrize("kind", ["ast", "code"])
def test_publication_archive_preserves_declared_parents(tmp_path, monkeypatch, kind):
    parents = ["fac_existing_parent", "generator-local-reference"]
    candidate = (FactorExpressionAST("close", parent_ids=parents) if kind == "ast"
                 else FactorExpressionCode("factor = df['close']", parent_ids=parents))
    if kind == "code":
        monkeypatch.setattr(candidate, "compute", lambda frame: frame["close"])
    director, storage = _publication_archive_director(tmp_path, monkeypatch, [candidate])
    assert director.run(1) == [candidate]
    factor_id, = director.get_latest_factor_ids()
    assert factor_id.startswith("fac_") and len(factor_id) == 12
    metadata = storage.get_metadata(factor_id)
    assert metadata.parent_ids == parents
    assert metadata.data_lineage["snapshot_status"] == "available"
    assert storage.load_factor_values(factor_id) is not None


@pytest.mark.parametrize("user_config,expected", [({}, "unknown"), ({"user_id": None}, "unknown"),
                                                ({"user_id": "researcher-7"}, "researcher-7")])
def test_publication_archive_author_is_configured_or_unattributed(tmp_path, monkeypatch, user_config, expected):
    candidate = FactorExpressionAST("close")
    director, storage = _publication_archive_director(tmp_path, monkeypatch, [candidate], **user_config)
    legacy = FactorMetadata("legacy-id", "GP", "historical-author", logic_hash="legacy-hash")
    storage.save_gp_factor("close", legacy)
    legacy_path = Path(storage.meta_dir) / "legacy-id.json"
    before = legacy_path.read_bytes()
    director.run(1)
    factor_id, = director.get_latest_factor_ids()
    assert storage.get_metadata(factor_id).user_id == expected
    assert storage.get_metadata("legacy-id").user_id == "historical-author"
    assert legacy_path.read_bytes() == before
    assert "legacy-hash" in storage.get_all_logic_hashes()


@pytest.mark.parametrize("model_config,model_override,response", [
    ({"model": "config-model", "model_env": "FM_ARCHIVE_MODEL"}, "resolved-model", "factor = df['close']"),
    ({}, None, "factor = df['close']"),
    ({"model": "config-model"}, None, None),
])
def test_publication_archive_llm_model_matches_generation_source(monkeypatch, model_config, model_override, response):
    import user_workspace.custom_miners.my_custom_llm as module
    from core.miner.paradigms.llm_api_manager import LLMAPIManager

    calls = []
    # Keep the real manager's default/resolved request model, replacing only transport.
    monkeypatch.setattr("core.miner.paradigms.llm_api_manager.aiohttp", mock.Mock())
    def generate(manager, prompts):
        calls.append(manager.model)
        return [response]
    monkeypatch.setattr(LLMAPIManager, "batch_generate", generate)
    monkeypatch.setattr(module, "LLMAPIManager", LLMAPIManager)
    monkeypatch.delenv("FM_ARCHIVE_MODEL", raising=False)
    if model_override is not None:
        monkeypatch.setenv("FM_ARCHIVE_MODEL", model_override)
    miner = MyCustomLLMMiner(pd.DataFrame({"close": [1., 2.]}), {
        "population_size": 1,
        "data_feeds": {"required_streams": ["close"]},
        "llm_api_config": {"keys": ["fixture-credential"], **model_config},
    })
    miner.initialize_search_space()
    monkeypatch.setenv("FM_ARCHIVE_MODEL", "changed-after-manager-init")
    candidate, = miner.generate_candidates()
    provenance = candidate.get_provenance()
    assert calls == [model_override or model_config.get("model", "gpt-4")]
    assert provenance["model"] == (calls[0] if response else "")
    assert provenance["generation_source"] == ("live_api" if response else "fallback")
    assert "fixture-credential" not in json.dumps(provenance)


@pytest.mark.parametrize("failure", ["false", "exception"])
def test_publication_archive_snapshot_failure_is_explicit_and_keeps_other_successes(tmp_path, monkeypatch, failure):
    candidates = [FactorExpressionAST(value) for value in (1, 2, 3)]
    for index, candidate in enumerate(candidates):
        candidate.logic_hash = f"hash-{index}"
    director, storage = _publication_archive_director(tmp_path, monkeypatch, candidates)
    original = storage.save_factor_snapshot
    attempted = []
    def save(factor_id, values, returns, lineage):
        attempted.append(factor_id)
        if len(attempted) == 2:
            if failure == "exception":
                raise OSError("fixture snapshot unavailable")
            return False
        return original(factor_id, values, returns, lineage)
    monkeypatch.setattr(storage, "save_factor_snapshot", save)
    with pytest.raises(RuntimeError, match="snapshot") as error:
        director.run(1)
    assert len(attempted) == 3
    assert director.get_latest_factor_ids() == [attempted[0], attempted[2]]
    assert all(factor_id in str(error.value) for factor_id in attempted)
    director._print_results_table.assert_not_called()
    failed = storage.get_metadata(attempted[1])
    assert failed.data_lineage["snapshot_status"] == "failed"
    assert failed.data_lineage["snapshot_error"]
    assert storage.load_factor_values(attempted[1]) is None
    assert storage.get_all_logic_hashes() == {"hash-0", "hash-2"}
    for factor_id in director.get_latest_factor_ids():
        assert storage.get_metadata(factor_id).data_lineage["snapshot_status"] == "available"


def test_publication_archive_compute_failure_is_not_an_available_snapshot(tmp_path, monkeypatch):
    candidate = FactorExpressionAST("close")
    director, storage = _publication_archive_director(tmp_path, monkeypatch, [candidate])
    monkeypatch.setattr(candidate, "compute", mock.Mock(side_effect=ValueError("fixture computation failed")))
    with pytest.raises(RuntimeError, match="fixture computation failed"):
        director.run(1)
    assert director.get_latest_factor_ids() == []
    metadata, = storage.list_metadata()
    assert metadata.data_lineage["snapshot_status"] == "failed"
    assert storage.load_factor_values(metadata.factor_id) is None


@pytest.mark.parametrize("existing", [False, True])
def test_publication_archive_partial_snapshot_write_preserves_previous_file(tmp_path, monkeypatch, existing):
    storage = LocalFactorStorage(str(tmp_path))
    storage.save_gp_factor("close", FactorMetadata("stable-id", "GP", "test", logic_hash="hash"))
    values = pd.Series([1., 2.], index=pd.date_range("2025-01-01", periods=2))
    path = Path(storage.val_dir) / "stable-id.parquet"
    if existing:
        assert storage.save_factor_snapshot("stable-id", values, values / 100, {})
    before = path.read_bytes() if existing else None
    def partial_write(frame, target, **kwargs):
        Path(target).write_bytes(b"partial parquet")
        raise OSError("fixture write failed")
    monkeypatch.setattr(pd.DataFrame, "to_parquet", partial_write)
    assert storage.save_factor_snapshot("stable-id", values + 1, values / 100, {}) is False
    if existing:
        assert path.read_bytes() == before
    else:
        assert not path.exists()
    assert list(Path(storage.val_dir).iterdir()) == ([path] if existing else [])
    assert storage.get_metadata("stable-id").data_lineage["snapshot_status"] == "failed"
    assert "snapshot_file" not in storage.get_metadata("stable-id").data_lineage
    assert storage.load_factor_values("stable-id") is None
    assert "hash" not in storage.get_all_logic_hashes()


def test_publication_archive_rejected_metadata_save_does_not_report_an_id(tmp_path, monkeypatch):
    candidate = FactorExpressionAST("close")
    director, storage = _publication_archive_director(tmp_path, monkeypatch, [candidate])
    monkeypatch.setattr(storage, "save_gp_factor", lambda *args: False)
    snapshot = mock.Mock()
    monkeypatch.setattr(storage, "save_factor_snapshot", snapshot)
    with pytest.raises(RuntimeError, match="metadata"):
        director.run(1)
    assert director.get_latest_factor_ids() == []
    snapshot.assert_not_called()


def test_publication_archive_failed_final_metadata_does_not_expose_values(tmp_path, monkeypatch):
    storage = LocalFactorStorage(str(tmp_path))
    storage.save_gp_factor("close", FactorMetadata("metadata-failure", "GP", "test", logic_hash="retryable"))
    original = storage._save_meta
    def save(metadata):
        if metadata.data_lineage.get("snapshot_status") == "available":
            raise OSError("fixture final metadata failed")
        original(metadata)
    monkeypatch.setattr(storage, "_save_meta", save)
    values = pd.Series([1., 2.], index=pd.date_range("2025-01-01", periods=2))
    assert storage.save_factor_snapshot("metadata-failure", values, values / 100, {}) is False
    assert (Path(storage.val_dir) / "metadata-failure.parquet").is_file()
    metadata = storage.get_metadata("metadata-failure")
    assert metadata.data_lineage["snapshot_status"] == "failed"
    assert "final metadata failed" in metadata.data_lineage["snapshot_error"]
    from api.main import _factor_detail
    detail = _factor_detail(metadata)
    assert "snapshot_error" not in detail["metadata"]["data_lineage"]
    assert "snapshot_error" not in detail["audit_snapshot"]["lineage"]
    assert "snapshot_error" in metadata.data_lineage
    assert storage.load_factor_values("metadata-failure") is None
    assert "retryable" not in storage.get_all_logic_hashes()


@pytest.mark.parametrize("status", [None, "pending", "failed"])
def test_publication_archive_api_does_not_advertise_unavailable_snapshot(tmp_path, monkeypatch, status):
    import api.main as api
    from fastapi import HTTPException

    monkeypatch.chdir(tmp_path)
    storage = LocalFactorStorage()
    storage.save_gp_factor("close", FactorMetadata("api-snapshot", "GP", "test"))
    values = pd.Series([1., 2.], index=pd.date_range("2025-01-01", periods=2))
    assert storage.save_factor_snapshot("api-snapshot", values, values / 100, {})
    metadata = storage.get_metadata("api-snapshot")
    if status is None:
        metadata.data_lineage.pop("snapshot_status")  # An existing legacy archive.
    else:
        # File and old snapshot fields may still exist after an interrupted update.
        metadata.data_lineage["snapshot_status"] = status
    storage._save_meta(metadata)
    assert api._factor_summary(metadata)["snapshot_available"] is (status is None)
    assert api._factor_detail(metadata)["audit_snapshot"]["values_available"] is (status is None)
    if status is not None:
        with pytest.raises(HTTPException) as error:
            api._factor_analysis_payload("api-snapshot")
        assert error.value.status_code == 409
    else:
        assert storage.load_factor_values("api-snapshot") is not None


@pytest.mark.parametrize("api_config,expected_model", [
    ({"keys": ["fixture-credential"], "model": "configured-model"}, "configured-model"),
    ({"keys": ["fixture-credential"]}, "gpt-4"),
    ({}, ""),
])
def test_publication_archive_builtin_llm_records_the_request_model(monkeypatch, api_config, expected_model):
    from core.miner.paradigms.llm_api_manager import LLMAPIManager
    from core.miner.paradigms.llm_miner import LLMFactorMiner

    monkeypatch.setattr("core.miner.paradigms.llm_api_manager.aiohttp", mock.Mock())
    calls = []
    def generate(manager, prompts):
        calls.append(manager.model)
        return ["factor = df['close']"]
    monkeypatch.setattr(LLMAPIManager, "batch_generate", generate)
    miner = LLMFactorMiner(pd.DataFrame({"close": [1., 2.], "volume": [3., 4.]}), {
        "batch_size": 1, "llm_api_config": dict(api_config),
    })
    miner.initialize_search_space()
    miner.config["llm_api_config"]["model"] = "changed-after-manager-init"
    candidates = miner.generate_candidates()
    assert candidates
    assert calls == ([expected_model] if api_config else [])
    for candidate in candidates:
        assert candidate.get_provenance()["model"] == expected_model
        assert candidate.get_provenance()["generation_source"] == ("live_api" if api_config else "mock")


def test_publication_archive_public_cli_reports_partial_ids_and_failure(tmp_path, monkeypatch, caplog):
    import logging
    import sys
    from core import cli
    from core.utils.dynamic_loader import ModuleLoadReport

    monkeypatch.chdir(tmp_path)
    candidates = [FactorExpressionAST(value) for value in (1, 2, 3)]
    director, storage = _publication_archive_director(tmp_path, monkeypatch, candidates)
    original = storage.save_factor_snapshot
    attempted = []
    def save(factor_id, *args):
        attempted.append(factor_id)
        return False if len(attempted) == 2 else original(factor_id, *args)
    monkeypatch.setattr(storage, "save_factor_snapshot", save)
    monkeypatch.setattr("core.commands.mine.FactorMinerDirector", lambda *args: director)
    monkeypatch.setattr("core.commands.mine.load_user_modules", lambda *args: ModuleLoadReport())
    monkeypatch.setattr("core.data_feed.real_client.RealDataClient", lambda *args: director.data_client)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"max_iterations": 1, "data_feeds": {
        "required_streams": ["close"], "pairs": ["BTC/USDT:USDT"],
    }}))
    monkeypatch.setattr(sys, "argv", ["factorminer", "mine", "--miner", "GP", "--config", str(config)])
    with caplog.at_level(logging.INFO), pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 1
    assert len(attempted) == 3
    assert director.get_latest_factor_ids() == [attempted[0], attempted[2]]
    assert "Factor archival incomplete" in caplog.text
    assert all(factor_id in caplog.text for factor_id in attempted)
    assert "FactorMiner execution completed successfully" not in caplog.text
    assert len(storage.list_metadata()) == 3
    assert storage.load_factor_values(attempted[0]) is not None
    assert storage.load_factor_values(attempted[1]) is None
    assert storage.load_factor_values(attempted[2]) is not None


if __name__ == "__main__":
    unittest.main()
