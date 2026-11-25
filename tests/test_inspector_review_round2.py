"""R3/R4/R8: UTC data identity and truthful Inspector CLI outcomes."""

import builtins
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from core.data_feed.naming import data_path
from core.data_feed.real_client import RealDataClient
from core.inspector.engine import FactorInspectorEngine
from core.inspector.metrics import InspectorMetricEngine
from core.inspector.reporter import InspectorReporter
from core.miner.entities import FactorMetadata
from core.miner.nn import NumpyMLPFactorModel
from core.storage.factor_storage import LocalFactorStorage


ROOT = Path(__file__).resolve().parents[1]
FUTURES_PAIR = "BTC/USDT:USDT"
SPOT_PAIR = "BTC/USDT"
DATES = pd.date_range("2024-01-01", periods=64, freq="min")


def market_frame(close=None, timezone=None, with_volume=True):
    time = np.arange(len(DATES))
    close = np.asarray(close) if close is not None else 100 + np.sin(time / 4) * 3 + time * .02
    stored_dates = DATES if timezone is None else DATES.tz_localize("UTC").tz_convert(timezone)
    result = pd.DataFrame({"date": stored_dates, "open": close + .2, "high": close + 1,
                           "low": close - 1, "close": close})
    if with_volume:
        result["volume"] = 10 + np.cos(time / 3)
    return result


def write_market(root, venue="binance", instrument="futures", pair=FUTURES_PAIR, **kwargs):
    path = data_path(root / "data", venue, pair, "1m", instrument)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = market_frame(**kwargs)
    frame.to_feather(path)
    return frame, path


def config(venue="binance", instrument="futures", pairs=None):
    return {"data_feeds": {"exchange": venue, "instrument_type": instrument,
                           "pairs": pairs or [FUTURES_PAIR], "timeframe": "1m",
                           "mining_mode": "sequential_single",
                           "mine_period": [["2024-01-01", "2024-01-01 01:03:00"]]}}


def save_nn(root, features=("close",), mode="sequential_single"):
    storage = LocalFactorStorage(str(root / "factor_db"))
    model = NumpyMLPFactorModel(list(features), 2, mode=mode, random_seed=19)
    scaler_data = np.asarray([[95., 9.], [100., 10.], [105., 11.]])[:, :len(features)]
    model.fit_scaler(scaler_data)
    artifact = model.export_artifact()
    filename = storage.save_model_artifact("round2_nn", artifact.payload, artifact.extension)
    metadata = FactorMetadata("fac_round2_nn", "NN", "test")
    storage.save_nn_factor_channel("round2_nn", 0, metadata, artifact.logic_reference(filename))
    return model


@pytest.mark.parametrize("timezone", [None, "UTC", "Asia/Shanghai", "America/New_York"])
@pytest.mark.parametrize("aware_bounds", [False, True])
def test_real_client_preserves_utc_slice_and_label_alignment(tmp_path, monkeypatch, timezone, aware_bounds):
    monkeypatch.chdir(tmp_path)
    original, path = write_market(tmp_path, timezone=timezone)
    before = path.read_bytes()
    cfg = config()
    cfg["target"] = {"entry_price": "next_open", "horizon_bars": 2}
    bounds = ["2024-01-01 00:05", "2024-01-01 00:24"]
    if aware_bounds:
        bounds = ["2024-01-01 08:05+08:00", "2024-01-01 08:24+08:00"]
    cfg["data_feeds"]["mine_period"] = [bounds]
    cfg["data_feeds"]["test_period"] = [["2024-01-01 00:30", "2024-01-01 00:49"]]
    client = RealDataClient(cfg)
    for data, returns, start, end in [
        (client.get_data(), client.get_returns(), 5, 25),
        (client.get_test_data(), client.get_test_returns(), 30, 50),
    ]:
        assert len(data) == 20
        assert data.index.tz is None
        assert data.index.equals(pd.DatetimeIndex(DATES[start:end], name="date"))
        np.testing.assert_allclose(data.close, original.close.iloc[start:end])
        expected = data.close.shift(-2).div(data.open.shift(-1)).sub(1.).rename("returns")
        pd.testing.assert_series_equal(returns, expected)
    assert path.read_bytes() == before


def test_cross_asset_reader_aligns_different_storage_timezones(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    other_pair = "ETH/USDT:USDT"
    write_market(tmp_path, timezone="UTC")
    write_market(tmp_path, pair=other_pair, timezone="Asia/Shanghai", close=200 + np.arange(64) ** 1.1)
    cfg = config(pairs=[FUTURES_PAIR, other_pair])
    cfg["data_feeds"]["mining_mode"] = "cross_asset"
    cfg["data_feeds"]["mine_period"] = [["2024-01-01", "2024-01-01 00:19"]]
    client = RealDataClient(cfg)
    close, returns = client.get_data()["close"], client.get_returns()
    assert close.shape == (20, 2)
    assert close.notna().all().all()
    assert close.index.equals(pd.DatetimeIndex(DATES[:20], name="date"))
    pd.testing.assert_frame_equal(returns, close.shift(-1).div(close).sub(1.))


def test_nn_inspector_uses_config_exchange_and_allows_explicit_override(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    first, _ = write_market(tmp_path, venue="okx")
    second, _ = write_market(tmp_path, close=100 + np.cos(np.arange(64) / 6) * 5 + np.arange(64) * .03)
    model = save_nn(tmp_path)
    engine = FactorInspectorEngine(config(venue="okx"))
    configured = engine.inspect(factor_id="fac_round2_nn")[FUTURES_PAIR]
    overridden = engine.inspect(factor_id="fac_round2_nn", exchange="binance")[FUTURES_PAIR]
    expected = []
    for frame in [first, second]:
        prices = frame.set_index("date")
        factor = model.predict_channel(prices, 0)
        labels = prices.close.shift(-1).div(prices.close).sub(1.)
        expected.append(factor.corr(labels, method="spearman"))
    assert configured["spearman"]["overall"] == pytest.approx(expected[0])
    assert overridden["spearman"]["overall"] == pytest.approx(expected[1])
    assert abs(expected[0] - expected[1]) > .01


def test_inspector_market_config_and_explicit_override_precedence(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    spot, _ = write_market(tmp_path, venue="okx", instrument="spot", pair=SPOT_PAIR)
    futures, _ = write_market(tmp_path, venue="okx", close=100 + np.cos(np.arange(64) / 6) * 5)
    engine = FactorInspectorEngine(config("okx", "spot", [SPOT_PAIR]))
    configured = engine.inspect(ast_str='"close"')[SPOT_PAIR]
    overridden = engine.inspect(ast_str='"close"', instrument_type="futures", pairs=[FUTURES_PAIR])[FUTURES_PAIR]
    for result, frame in [(configured, spot), (overridden, futures)]:
        assert "error" not in result
        labels = frame.close.shift(-1).div(frame.close).sub(1.)
        assert result["spearman"]["overall"] == pytest.approx(frame.close.corr(labels, method="spearman"))


def test_inspector_uses_binance_futures_when_identity_is_omitted(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    frame, _ = write_market(tmp_path)
    cfg = config()
    cfg["data_feeds"].pop("exchange")
    cfg["data_feeds"].pop("instrument_type")
    result = FactorInspectorEngine(cfg).inspect(ast_str='"close"')[FUTURES_PAIR]
    assert "error" not in result
    labels = frame.close.shift(-1).div(frame.close).sub(1.)
    assert result["spearman"]["overall"] == pytest.approx(frame.close.corr(labels, method="spearman"))


def run_cli(root, cfg, *, factor=True):
    cfg_path = root / "inspect_config.json"
    cfg_path.write_text(json.dumps(cfg))
    workspace = root / "empty_workspace"
    workspace.mkdir(exist_ok=True)
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT), NO_COLOR="1", COLUMNS="160")
    args = [sys.executable, "-B", "-m", "core.cli", "inspect", "--config", str(cfg_path),
            "--user-dir", str(workspace)]
    args.extend(["--factor", "fac_round2_nn"] if factor else ["--ast", '"close"'])
    return subprocess.run(args, cwd=root, env=environment, capture_output=True, text=True, timeout=20)


@pytest.mark.parametrize("with_volume,failed", [([False, False], 2), ([False, True], 1), ([True, True], 0)])
def test_public_nn_inspector_cli_reports_each_failure_and_exit_status(tmp_path, with_volume, failed):
    pairs = [FUTURES_PAIR, "ETH/USDT:USDT"]
    for pair, complete in zip(pairs, with_volume):
        write_market(tmp_path, pair=pair, with_volume=complete)
    save_nn(tmp_path, features=("close", "volume"))
    result = run_cli(tmp_path, config(pairs=pairs))
    assert result.returncode == (1 if failed else 0), result.stderr
    assert result.stdout.count("Inspection unavailable") == failed
    assert result.stdout.count("Overall RankIC") == 2 - failed
    if failed:
        assert "missing required features" in result.stdout
        assert "volume" in result.stdout
        assert f"Inspection failed for {failed} of 2 requested pairs" in result.stderr
        assert FUTURES_PAIR in result.stdout
    else:
        assert "Inspection failed" not in result.stderr


def test_public_cli_continues_after_one_unreadable_file(tmp_path):
    pairs = [FUTURES_PAIR, "ETH/USDT:USDT"]
    _, corrupt_path = write_market(tmp_path)
    corrupt_path.write_bytes(b"invalid fixture feather")
    write_market(tmp_path, pair=pairs[1])
    result = run_cli(tmp_path, config(pairs=pairs), factor=False)
    assert result.returncode == 1
    assert result.stdout.count("Inspection unavailable") == 1
    assert result.stdout.count("Overall RankIC") == 1
    assert "Inspection failed for 1 of 2 requested pairs" in result.stderr


def test_public_cli_empty_slice_is_unavailable_without_zero_metrics(tmp_path):
    write_market(tmp_path)
    cfg = config()
    cfg["data_feeds"]["mine_period"] = [["2024-02-01", "2024-02-02"]]
    result = run_cli(tmp_path, cfg, factor=False)
    assert result.returncode == 1
    assert "Inspection unavailable" in result.stdout
    assert "No data available" in result.stdout
    assert "Overall RankIC" not in result.stdout


def test_public_nn_cli_all_nan_output_is_unavailable(tmp_path):
    frame, path = write_market(tmp_path)
    frame["volume"] = np.nan
    frame.to_feather(path)
    save_nn(tmp_path, features=("close", "volume"))
    result = run_cli(tmp_path, config())
    assert result.returncode == 1
    assert "Inspection unavailable" in result.stdout
    assert "at least one finite value" in result.stdout
    assert "Overall RankIC" not in result.stdout


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_public_ast_cli_rejects_nonfinite_input(tmp_path, value):
    frame, path = write_market(tmp_path)
    frame.loc[5, "close"] = value
    frame.to_feather(path)
    result = run_cli(tmp_path, config(), factor=False)
    assert result.returncode == 1
    assert "Inspection unavailable" in result.stdout
    assert "must not contain infinite values" in result.stdout
    assert "Overall RankIC" not in result.stdout


@pytest.mark.parametrize("bars,valid", [(1, 0), (6, 5), (10, 9)])
def test_public_cli_insufficient_joint_samples_are_unavailable(tmp_path, bars, valid):
    write_market(tmp_path)
    cfg = config()
    cfg["data_feeds"]["mine_period"] = [[str(DATES[0]), str(DATES[bars - 1])]]
    result = run_cli(tmp_path, cfg, factor=False)
    assert result.returncode == 1
    assert "Inspection unavailable" in result.stdout
    assert f"{valid}/{bars}" in result.stdout
    assert "at least 10" in result.stdout
    assert "Overall RankIC" not in result.stdout


def test_public_cli_minimum_joint_sample_count_is_still_available(tmp_path):
    write_market(tmp_path)
    cfg = config()
    cfg["data_feeds"]["mine_period"] = [[str(DATES[0]), str(DATES[10])]]
    result = run_cli(tmp_path, cfg, factor=False)
    assert result.returncode == 0, result.stderr
    assert "10 / 11" in result.stdout
    assert "Overall RankIC" in result.stdout


def test_inspector_rejects_silent_output_index_alignment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    write_market(tmp_path)
    engine = FactorInspectorEngine(config())
    class MisalignedExpression:
        def to_display_string(self):
            return "misaligned fixture"
        def compute(self, data):
            result = data.close.copy()
            result.index = result.index + pd.Timedelta(minutes=1)
            return result
    monkeypatch.setattr(engine.resolver, "resolve", lambda **kwargs: MisalignedExpression())
    result = engine.inspect(ast_dict="close")[FUTURES_PAIR]
    assert "indexes must match exactly" in result["error"]
    assert "spearman" not in result


def test_plain_text_reporter_keeps_error_distinct_from_zero(monkeypatch, capsys):
    original_import = builtins.__import__
    def without_rich(name, *args, **kwargs):
        if name == "rich" or name.startswith("rich."):
            raise ImportError("fixture without Rich")
        return original_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", without_rich)
    InspectorReporter.print_inspection_report("fixture", {
        "FAILED": {"error": "missing required input"},
        "ZERO": {"spearman": {"overall": 0., "ir": 0., "t_stat": 0.}},
    })
    output = capsys.readouterr().out
    failed_section, zero_section = output.split("Pair: ZERO")
    assert "Inspection unavailable: missing required input" in failed_section
    assert "RankIC Overall" not in failed_section
    assert "RankIC Overall: 0.0000" in zero_section


@pytest.mark.parametrize("split,segment_bars,target,aware_override", [
    ("disjoint", 6, {}, False),
    ("adjacent", 16, {"entry_price": "next_open", "horizon_bars": 3,
                      "return_type": "log"}, True),
    ("continuous", 32, {}, False),
])
def test_inspector_lag_keeps_requested_period_boundaries(
    tmp_path, monkeypatch, split, segment_bars, target, aware_override,
):
    monkeypatch.chdir(tmp_path)
    first = pd.date_range("2024-01-01", periods=segment_bars, freq="min")
    if split == "continuous":
        dates = first
        close = 100. + np.arange(segment_bars)
        slices = [slice(0, segment_bars)]
    else:
        start = "2024-01-10" if split == "disjoint" else first[-1] + pd.Timedelta(minutes=1)
        second = pd.date_range(start, periods=segment_bars, freq="min")
        dates = first.append(second)
        close = np.r_[100. + np.arange(segment_bars), 1000. + np.arange(segment_bars)]
        slices = [slice(0, segment_bars), slice(segment_bars, 2 * segment_bars)]
    frame = pd.DataFrame({"date": dates, "open": close - .25, "high": close + 1,
                          "low": close - 1, "close": close,
                          "volume": 1. + np.arange(len(dates))})
    _, path = write_market(tmp_path)
    frame.to_feather(path)
    data = frame.set_index("date")
    periods = [[str(dates[part][0]), str(dates[part][-1])] for part in slices]
    if aware_override:
        periods = [[str(pd.Timestamp(value).tz_localize("UTC").tz_convert("Asia/Shanghai"))
                    for value in period] for period in periods]
    cfg = config()
    cfg["target"] = target
    if not aware_override:
        cfg["data_feeds"]["mine_period"] = periods
    kwargs = {"periods": periods} if aware_override else {}
    result = FactorInspectorEngine(cfg).inspect(ast_dict="volume", **kwargs)[FUTURES_PAIR]
    assert "error" not in result

    def expected_labels(lag):
        labels = []
        for part in slices:
            segment = data.iloc[part]
            entry = segment.open.shift(-1) if target.get("entry_price") == "next_open" else segment.close
            ratio = segment.close.shift(-lag).div(entry)
            labels.append(np.log(ratio) if target.get("return_type") == "log" else ratio - 1.)
        return pd.concat(labels)

    labels = expected_labels(target.get("horizon_bars", 1))
    assert result["spearman"]["overall"] == pytest.approx(data.volume.corr(labels, method="spearman"))
    assert result["valid_bars"] == labels.notna().sum()
    assert result["coverage"] == pytest.approx(labels.notna().mean())
    for lag in [1, 2, 3, 5, 10]:
        lag_labels = expected_labels(lag)
        expected = data.volume.corr(lag_labels, method="spearman") if lag_labels.notna().sum() >= 10 else 0.
        assert result["decay"][f"Lag_{lag}"] == pytest.approx(expected), f"{split}, lag={lag}"


@pytest.mark.parametrize("constant", [False, True])
def test_inspector_missing_quantile_groups_are_unavailable(tmp_path, monkeypatch, constant):
    monkeypatch.chdir(tmp_path)
    frame, path = write_market(tmp_path, close=100. * 1.01 ** np.arange(64))
    frame["volume"] = 0. if constant else np.r_[np.zeros(10), np.ones(54)]
    frame.to_feather(path)
    cfg = config()
    cfg["data_feeds"]["mine_period"] = [[str(DATES[0]), str(DATES[20])]]
    result = FactorInspectorEngine(cfg).inspect(ast_dict="volume")[FUTURES_PAIR]
    assert "error" not in result
    assert result["valid_bars"] == 20
    quantiles = result["quantiles"]
    if constant:
        assert quantiles["Q1"] is None
    else:
        assert quantiles["Q1"] == pytest.approx(.01)
    for key in ["Q2", "Q3", "Q4", "Q5", "Long_Short"]:
        assert quantiles[key] is None


def test_inspector_complete_quantiles_preserve_real_zero_returns():
    factor = pd.Series(np.arange(20, dtype=float))
    returns = pd.Series(np.repeat([0., .01, .02, .03, .04], 4))
    result = InspectorMetricEngine.calculate_comprehensive_metrics(factor, returns)
    assert result["quantiles"] == pytest.approx({
        "Q1": 0., "Q2": .01, "Q3": .02, "Q4": .03, "Q5": .04, "Long_Short": .04,
    })


def test_inspector_quantile_failure_is_unavailable(monkeypatch):
    def fail_qcut(*args, **kwargs):
        raise ValueError("synthetic grouping failure")
    monkeypatch.setattr(pd, "qcut", fail_qcut)
    factor = pd.Series(np.arange(20, dtype=float))
    result = InspectorMetricEngine.calculate_comprehensive_metrics(factor, factor * .001)
    assert result["quantiles"] == dict.fromkeys(["Q1", "Q2", "Q3", "Q4", "Q5", "Long_Short"])


@pytest.mark.parametrize("plain_text", [False, True])
def test_inspector_reporter_shows_unavailable_quantiles_without_hiding_zero(
    monkeypatch, capsys, plain_text,
):
    monkeypatch.setenv("COLUMNS", "200")
    if plain_text:
        original_import = builtins.__import__
        def without_rich(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("fixture without Rich")
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", without_rich)
    InspectorReporter.print_inspection_report("fixture", {"SYNTHETIC": {
        "quantiles": {"Q1": .01, "Q2": None, "Q3": 0., "Q4": .02,
                      "Q5": None, "Long_Short": None},
    }})
    output = capsys.readouterr().out
    assert "Q1: 100.0bps" in output
    assert "Q2: N/A" in output
    assert "Q3: 0.0bps" in output
    assert "Q5: N/A" in output
    spread_line = next(line for line in output.splitlines() if "Long-Short Spread" in line)
    assert "N/A" in spread_line
    assert "+0.0" not in spread_line


@pytest.mark.parametrize("plain_text", [False, True])
def test_cross_asset_unimplemented_quantiles_are_unavailable_in_reports(
    monkeypatch, capsys, plain_text,
):
    factor = pd.DataFrame(np.arange(12)[:, None] + np.array([1., 2., 3.]),
                          columns=["A", "B", "C"])
    returns = factor * .01
    result = InspectorMetricEngine.calculate_comprehensive_metrics(factor, returns, lags=[1, 3])
    # Every cross section has identical ordering and linear returns. Each
    # factor advances by one per row; those existing metrics stay unchanged.
    assert result["total_bars"] == result["valid_bars"] == 36
    assert result["coverage"] == 1.
    assert result["turnover"] == 1.
    for key in ["pearson", "spearman"]:
        assert result[key] == pytest.approx({"overall": 1., "mean": 1., "std": 0.,
                                             "ir": 0., "t_stat": 0., "pos_ratio": 1.})
    assert result["decay"] == pytest.approx({"Lag_1": 1., "Lag_3": 1.})
    assert result["quantiles"] == {"Q1": None, "Q5": None, "Long_Short": None}

    monkeypatch.setenv("COLUMNS", "200")
    if plain_text:
        original_import = builtins.__import__
        def without_rich(name, *args, **kwargs):
            if name == "rich" or name.startswith("rich."):
                raise ImportError("fixture without Rich")
            return original_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", without_rich)
    InspectorReporter.print_inspection_report("cross-asset fixture", {"SYNTHETIC": result})
    output = capsys.readouterr().out
    assert "Q1: N/A" in output
    assert "Q5: N/A" in output
    assert "Q1: 0.0" not in output
    assert "Q5: 0.0" not in output
    spread_line = next(line for line in output.splitlines() if "Long-Short Spread" in line)
    assert "N/A" in spread_line
    assert "+0.0" not in spread_line
