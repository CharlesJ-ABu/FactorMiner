"""Regression coverage for review items 1, 10, 11, 12 and bounded item 13."""

import json
import sys
from types import ModuleType
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from core.execution.compiler import FactorCompiler
from core.data_feed.naming import data_path
from core.inspector.engine import FactorInspectorEngine
from core.inspector.metrics import InspectorMetricEngine
from core.inspector.resolver import FactorResolver
from core.miner.entities import FactorMetadata
from core.miner.expressions import FactorExpressionAST
from core.miner.nn import NumpyMLPFactorModel
from core.miner.operator_runtime import OperatorRuntimeError, evaluate_ast
from core.miner.registry import OperatorRegistry
from core.startup_validation import StartupValidationError, validate_mining_startup
from core.storage.factor_storage import LocalFactorStorage
from core.utils.dynamic_loader import load_user_modules


def test_dynamic_loader_registers_dataclass_module(tmp_path, monkeypatch):
    folder = tmp_path / "custom_miners"
    folder.mkdir()
    name = "custom_miners.review_dataclass_fixture"
    monkeypatch.setitem(sys.modules, name, None)
    (folder / "review_dataclass_fixture.py").write_text(
        "from __future__ import annotations\nfrom dataclasses import dataclass\n"
        "@dataclass\nclass Sample:\n    value: int\n"
    )
    report = load_user_modules(str(tmp_path))
    assert report.ok
    assert name in report.loaded_modules
    assert sys.modules[name].Sample(3).value == 3


@pytest.mark.parametrize("has_previous", [False, True])
def test_dynamic_loader_failure_restores_module_and_blocks_startup(tmp_path, monkeypatch, has_previous):
    folder = tmp_path / "custom_miners"
    folder.mkdir()
    name = "custom_miners.review_failure_fixture"
    previous = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, previous)
    if not has_previous:
        del sys.modules[name]
    (folder / "review_failure_fixture.py").write_text("raise RuntimeError('fixture import failure')\n")
    report = load_user_modules(str(tmp_path))
    assert len(report.errors) == 1
    if has_previous:
        assert sys.modules[name] is previous
    else:
        assert name not in sys.modules
    config = {"paradigm": "GP", "data_feeds": {"pairs": ["A"], "required_streams": ["close"]}}
    with pytest.raises(StartupValidationError, match="fixture import failure"):
        validate_mining_startup(config, report)


@pytest.mark.parametrize("import_fails", [False, True])
def test_public_inspector_cli_requires_complete_user_module_import(
    tmp_path, monkeypatch, capsys, caplog, import_fails
):
    from core import cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(OperatorRegistry, "_registry", dict(OperatorRegistry._registry))
    module_name = "custom_operators.review_inspector_startup"
    monkeypatch.setitem(sys.modules, module_name, ModuleType(module_name))
    workspace = tmp_path / "user_workspace"
    operators = workspace / "custom_operators"
    operators.mkdir(parents=True)
    executed = tmp_path / "operator_executed"
    source = (
        "from pathlib import Path\n"
        "from core.miner.registry import OperatorRegistry\n"
        "@OperatorRegistry.register(arity=1)\n"
        "def review_inspector_signal(value):\n"
        f"    Path({str(executed)!r}).write_text('executed')\n"
        "    return value.diff()\n"
    )
    if import_fails:
        source += "raise RuntimeError('fixture failed after operator registration')\n"
    (operators / "review_inspector_startup.py").write_text(source)

    pair = "BTC/USDT:USDT"
    target = data_path("data", "okx", pair, "1m", "futures")
    target.parent.mkdir(parents=True)
    close = 100 + np.arange(64) * .1 + np.sin(np.arange(64))
    pd.DataFrame({"date": pd.date_range("2024-01-01", periods=64, freq="min"),
                  "open": close, "high": close + 1, "low": close - 1,
                  "close": close, "volume": 10.}).to_feather(target)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"data_feeds": {
        "exchange": "okx", "instrument_type": "futures", "pairs": [pair],
        "timeframe": "1m", "mine_period": [["2024-01-01", "2024-01-02"]],
    }}))
    read_calls = []
    read_feather = pd.read_feather

    def record_read(*args, **kwargs):
        read_calls.append(args[0])
        return read_feather(*args, **kwargs)

    monkeypatch.setattr(pd, "read_feather", record_read)
    monkeypatch.setattr(sys, "argv", ["factorminer", "inspect", "--ast",
        json.dumps({"op": "review_inspector_signal", "left": "close"}),
        "--config", str(config), "--user-dir", str(workspace)])
    if import_fails:
        with pytest.raises(SystemExit) as stopped:
            cli.main()
        assert stopped.value.code == 1
        assert "Inspection startup failed" in caplog.text
        assert "fixture failed after operator registration" in caplog.text
        assert read_calls == []
        assert not executed.exists()
        assert "Overall RankIC" not in capsys.readouterr().out
    else:
        cli.main()
        assert len(read_calls) == 1
        assert executed.exists()
        assert "Overall RankIC" in capsys.readouterr().out


@pytest.mark.parametrize("cross_asset", [False, True])
def test_resolver_accepts_direct_zero_ast_terminal(tmp_path, cross_asset):
    resolver = FactorResolver(LocalFactorStorage(str(tmp_path)))
    frame = pd.DataFrame({"A": [1.0, 2.0]}, index=pd.date_range("2024-01-01", periods=2))
    data = {"close": frame} if cross_asset else frame.rename(columns={"A": "close"})
    values = resolver.resolve(ast_dict=0).compute(data)
    if cross_asset:
        pd.testing.assert_frame_equal(values, pd.DataFrame(0.0, index=frame.index, columns=frame.columns))
    else:
        pd.testing.assert_series_equal(values, pd.Series(0.0, index=frame.index))


@pytest.mark.parametrize("failure", ["missing", "empty_path", "invalid_json", "array", "null", "unreadable", "invalid_encoding"])
def test_public_inspector_cli_rejects_invalid_explicit_config_before_engine(
    tmp_path, monkeypatch, caplog, failure
):
    from core import cli

    workspace = tmp_path / "user_workspace"
    workspace.mkdir()
    config = tmp_path / "config.json"
    if failure == "invalid_json":
        config.write_text("{broken")
    elif failure in {"array", "null"}:
        config.write_text("[]" if failure == "array" else "null")
    elif failure == "unreadable":
        config.mkdir()
    elif failure == "invalid_encoding":
        config.write_bytes(b"\xff")
    config_arg = "" if failure == "empty_path" else str(config)
    engine = Mock(side_effect=AssertionError("invalid config must not construct an engine"))
    monkeypatch.setattr("core.commands.inspect.FactorInspectorEngine", engine)
    monkeypatch.setattr(sys, "argv", ["factorminer", "inspect", "--ast", '"close"',
        "--config", config_arg, "--user-dir", str(workspace)])

    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 1
    engine.assert_not_called()
    assert repr(config_arg) in caplog.text
    assert "Failed to load base config" in caplog.text or "must be a JSON object" in caplog.text


@pytest.mark.parametrize("provide_config", [False, True])
def test_public_inspector_cli_preserves_omitted_and_valid_config(tmp_path, monkeypatch, provide_config):
    from core import cli

    workspace = tmp_path / "user_workspace"
    workspace.mkdir()
    expected = {"data_feeds": {"exchange": "okx", "instrument_type": "spot"}} if provide_config else {}
    arguments = ["factorminer", "inspect", "--ast", '"close"', "--user-dir", str(workspace)]
    if provide_config:
        config = tmp_path / "config.json"
        config.write_text(json.dumps(expected))
        arguments.extend(["--config", str(config)])
    engine = Mock()
    engine.return_value.inspect.return_value = {}
    monkeypatch.setattr("core.commands.inspect.FactorInspectorEngine", engine)
    monkeypatch.setattr(sys, "argv", arguments)

    cli.main()
    engine.assert_called_once_with(base_config=expected)
    engine.return_value.inspect.assert_called_once()


@pytest.mark.parametrize("has_previous", [False, True])
@pytest.mark.parametrize("error_type", [SystemExit, KeyboardInterrupt])
def test_dynamic_loader_restores_module_before_propagating_base_exception(
    tmp_path, monkeypatch, has_previous, error_type
):
    folder = tmp_path / "custom_miners"
    folder.mkdir()
    name = "custom_miners.review_exit_fixture"
    previous = ModuleType(name)
    monkeypatch.setitem(sys.modules, name, previous)
    if not has_previous:
        del sys.modules[name]
    holder = ModuleType("review_exception_holder")
    holder.error = error_type(17)
    monkeypatch.setitem(sys.modules, holder.__name__, holder)
    (folder / "review_exit_fixture.py").write_text(
        "from review_exception_holder import error\npartial = True\nraise error\n"
    )
    with pytest.raises(error_type) as raised:
        load_user_modules(str(tmp_path))
    assert raised.value is holder.error
    if has_previous:
        assert sys.modules[name] is previous
    else:
        assert name not in sys.modules


@pytest.mark.parametrize("cross_asset", [False, True])
def test_missing_ast_stream_raises_while_zero_and_warmup_are_preserved(cross_asset):
    series = pd.Series([np.nan, 2.0, 3.0], name="close")
    data = {"close": series.to_frame("A")} if cross_asset else series.to_frame()
    with pytest.raises(OperatorRuntimeError, match="Missing required input stream 'missing_feature'"):
        evaluate_ast({"op": "add", "left": "close", "right": "missing_feature"}, data)
    actual = evaluate_ast({"op": "add", "left": "close", "right": 0}, data)
    expected = data["close"]
    if cross_asset:
        pd.testing.assert_frame_equal(actual, expected)
    else:
        pd.testing.assert_series_equal(actual, expected, check_names=False)


@pytest.mark.parametrize("node", ["close", 0, 2.5, {"op": "add", "left": "close", "right": 1}])
def test_compiler_round_trips_terminal_and_compound_ast(tmp_path, node):
    storage = LocalFactorStorage(str(tmp_path))
    storage.save_gp_factor(node, FactorMetadata("fac_ast", "MyCustomGP", "test"))
    data = pd.DataFrame({"close": [1.0, 2.0, 4.0]})
    actual = FactorCompiler(storage).compile_for_live_trading("fac_ast")(data)
    pd.testing.assert_series_equal(actual, FactorExpressionAST(node).compute(data))


@pytest.mark.parametrize("node", [None, [], {}, {"op": "add", "left": "close"}])
def test_compiler_rejects_malformed_stored_ast(tmp_path, node):
    storage = LocalFactorStorage(str(tmp_path))
    storage.save_gp_factor(node, FactorMetadata("fac_bad", "GP", "test"))
    with pytest.raises(OperatorRuntimeError):
        FactorCompiler(storage).compile_for_live_trading("fac_bad")


def test_rolling_spearman_uses_window_ranks_with_ties_and_partial_windows():
    rng = np.random.default_rng(42)
    factor = pd.Series(rng.integers(-5, 6, size=100), dtype=float)
    returns = pd.Series(rng.integers(-4, 5, size=100), dtype=float)
    expected = pd.Series([
        np.nan if end < 9 else spearmanr(factor.iloc[max(0, end-19):end+1],
                                       returns.iloc[max(0, end-19):end+1]).statistic
        for end in range(100)
    ])
    rolling = InspectorMetricEngine._rolling_spearman(factor, returns)
    pd.testing.assert_series_equal(rolling, expected)
    metrics = InspectorMetricEngine.calculate_comprehensive_metrics(factor, returns, lags=[])
    assert metrics["spearman"]["mean"] == pytest.approx(expected.mean())
    assert metrics["spearman"]["std"] == pytest.approx(expected.std())
    assert metrics["spearman"]["overall"] == pytest.approx(spearmanr(factor, returns).statistic)
    pd.testing.assert_series_equal(
        rolling.iloc[:40], InspectorMetricEngine._rolling_spearman(factor.iloc[:40], returns.iloc[:40])
    )


def test_rolling_spearman_handles_nan_and_constant_windows():
    factor = pd.Series([1.0] * 20 + list(range(20)), dtype=float)
    returns = pd.Series(np.arange(40), dtype=float)
    factor.iloc[23] = np.nan
    rolling = InspectorMetricEngine._rolling_spearman(factor, returns)
    assert rolling.iloc[:20].isna().all()
    assert rolling.iloc[-1] == pytest.approx(1.0)


def save_nn(storage, mode="sequential_single"):
    model = NumpyMLPFactorModel(["close", "volume"], 2, mode=mode, random_seed=7)
    model.fit_scaler(np.asarray([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]))
    artifact = model.export_artifact()
    model_version = f"nn_review_{mode}"
    filename = storage.save_model_artifact(model_version, artifact.payload, artifact.extension)
    metadata = FactorMetadata("fac_nn", "NN", "test")
    storage.save_nn_factor_channel(model_version, 1, metadata, artifact.logic_reference(filename))
    return model


@pytest.mark.parametrize("mode", ["sequential_single", "cross_asset"])
def test_nn_resolver_reproduces_saved_channel_predictions(tmp_path, mode):
    storage = LocalFactorStorage(str(tmp_path))
    model = save_nn(storage, mode)
    data = pd.DataFrame({"close": [1.5, 2.5], "volume": [15.0, 25.0]})
    if mode == "cross_asset":
        data = {key: data[key].to_frame("A") for key in data.columns}
    actual = FactorResolver(storage).resolve(factor_id="fac_nn").compute(data)
    expected = model.predict_channel(data, 1)
    if mode == "cross_asset":
        pd.testing.assert_frame_equal(actual, expected)
    else:
        pd.testing.assert_series_equal(actual, expected)


@pytest.mark.parametrize("failure", ["legacy", "unknown_format", "missing_file", "bad_channel", "missing_feature"])
def test_nn_resolver_fails_explicitly_for_incomplete_inputs(tmp_path, failure):
    storage = LocalFactorStorage(str(tmp_path))
    save_nn(storage)
    metadata = storage.get_metadata("fac_nn")
    ref = metadata.logic_reference.copy()
    expected_error = {
        "legacy": "legacy artifact", "unknown_format": "Unknown NN model format",
        "missing_file": "absent", "bad_channel": "valid saved channel",
        "missing_feature": "missing required features",
    }[failure]
    if failure == "legacy":
        ref.pop("model_format")
    elif failure == "unknown_format":
        ref["model_format"] = "not_registered_review_fixture"
    elif failure == "missing_file":
        ref["model_file"] = "absent.npz"
    channel = -1 if failure == "bad_channel" else 1
    ref.pop("channel")
    storage.save_nn_factor_channel("nn_review", channel, metadata, ref)
    with pytest.raises((ValueError, FileNotFoundError), match=expected_error):
        expression = FactorResolver(storage).resolve(factor_id="fac_nn")
        expression.compute(pd.DataFrame({"close": [1.0, 2.0]}))


def test_cli_inspector_nn_runs_sequential_and_rejects_cross_asset(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    engine = FactorInspectorEngine()
    save_nn(engine.resolver.storage)
    dates = pd.date_range("2024-01-01", periods=32, freq="min")
    data = pd.DataFrame({"close": 1 + np.arange(32) * .05 + .01 * np.sin(np.arange(32)),
                         "volume": np.arange(32) * .5 + 10}, index=dates)
    class Client:
        def __init__(self, config):
            assert config["data_feeds"]["mining_mode"] == "sequential_single"
        def get_data(self):
            return data
        def get_returns(self):
            return data.close.pct_change().shift(-1)
    monkeypatch.setattr("core.inspector.engine.RealDataClient", Client)
    monkeypatch.setattr("core.inspector.engine.InspectorReporter.print_inspection_report", lambda *args: None)
    result = engine.inspect(factor_id="fac_nn", pairs=["A"])["A"]
    assert "error" not in result
    assert result["valid_bars"] == 31
    assert "spearman" in result
    save_nn(engine.resolver.storage, "cross_asset")
    with pytest.raises(ValueError, match="cannot re-evaluate a cross_asset NN"):
        engine.inspect(factor_id="fac_nn", pairs=["A"])
