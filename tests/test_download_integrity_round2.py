"""Offline acceptance of requested coverage, OHLC validation and atomic saves.

Only the external exchange and rate-limit sleep are replaced. The health gate,
pagination, date normalization and Feather persistence all execute normally.
"""

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import sys
import threading
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from core import cli
from core.data_feed.batch_downloader import SmartBatchDownloader
from core.data_feed.data_downloader import DataDownloader, DataGapFillError
from core.data_feed.health_checker import health_checker
from core.data_feed.naming import data_path
from core.data_feed.real_client import RealDataClient


PAIR = "BTC/USDT:USDT"
START = pd.Timestamp("2024-01-01", tz="UTC")
START_MS = int(START.timestamp() * 1000)


def rows(count=1440):
    return [[START_MS + i * 60000, 100., 101., 99., 100., 10.] for i in range(count)]


def frame(count=20):
    result = pd.DataFrame(rows(count), columns=["date", "open", "high", "low", "close", "volume"])
    result["date"] = pd.to_datetime(result.date, unit="ms")
    return result


def client_for(candles, page_size=1000):
    client = Mock(id="okx", rateLimit=0)
    client.parse_timeframe.return_value = 60
    client.fetch_ohlcv.side_effect = lambda symbol, timeframe, since, limit: [
        row for row in candles if row[0] >= since
    ][:min(limit, page_size)]
    return client


def hourly_market():
    return [[START_MS + i * 3600000, 100., 101., 99., 100., 10.] for i in range(72)]


def seed_middle_day(target):
    target.parent.mkdir(parents=True, exist_ok=True)
    data = pd.DataFrame(hourly_market()[24:48], columns=["date", "open", "high", "low", "close", "volume"])
    data["date"] = pd.to_datetime(data.date, unit="ms")
    data.to_feather(target)


def download_hour_day(downloader, day, **identity):
    args = dict(exchange_id="okx", symbol=PAIR, timeframe="1h", trade_type="futures",
                start_date=f"2024-01-{day:02}", end_date=f"2024-01-{day + 1:02}")
    args.update(identity)
    entry = downloader.download_ohlcv_batch if isinstance(downloader, SmartBatchDownloader) else downloader.download_ohlcv
    return entry(**args)


@pytest.mark.parametrize("kinds", [
    (DataDownloader, DataDownloader), (SmartBatchDownloader, SmartBatchDownloader),
    (DataDownloader, SmartBatchDownloader),
])
def test_parallel_merge_preserves_both_successful_intervals(tmp_path, monkeypatch, kinds):
    monkeypatch.chdir(tmp_path)
    target = data_path("data", "okx", PAIR, "1h", "futures")
    seed_middle_day(target)
    if kinds[0] is not kinds[1]:
        # Different spellings of the same canonical absolute target share a lock.
        monkeypatch.setattr("core.data_feed.batch_downloader.data_path", lambda *args:
            data_path(*args).parent / ".." / data_path(*args).parent.name / data_path(*args).name)
    clients = [kind() for kind in kinds]
    for downloader in clients:
        exchange = client_for(hourly_market())
        exchange.parse_timeframe.return_value = 3600
        monkeypatch.setattr(downloader, "get_exchange_instance", lambda *args, _exchange=exchange, **kwargs: _exchange)
    monkeypatch.setattr("core.data_feed.data_downloader.time.sleep", lambda _: None)
    first_at_save, second_at_save, first_saved = (threading.Event() for _ in range(3))
    local = threading.local()
    real_write = DataDownloader._write_feather_atomic
    observed_first = []

    def ordered_write(data, path):
        if local.name == "A":
            first_at_save.set()
            # Without the transaction lock B reaches save with its stale frame.
            # With the lock B waits before reading; A proceeds after this bounded
            # scheduling window. Correctness is asserted on saved rows, not time.
            second_at_save.wait(.5)
            real_write(data, path)
            observed_first.extend(pd.read_feather(path).date.tolist())
            first_saved.set()
        else:
            second_at_save.set()
            assert first_saved.wait(2)
            real_write(data, path)

    monkeypatch.setattr(DataDownloader, "_write_feather_atomic", staticmethod(ordered_write))
    def run(index, day, name):
        local.name = name
        return download_hour_day(clients[index], day)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(run, 0, 1, "A")
        assert first_at_save.wait(2)
        second = pool.submit(run, 1, 3, "B")
        outcomes = [first.result(timeout=3), second.result(timeout=3)]
    assert all(outcome["success"] for outcome in outcomes), outcomes
    saved = pd.read_feather(target)
    expected = list(pd.date_range("2024-01-01", periods=72, freq="h"))
    assert len(observed_first) == 48
    assert saved.date.tolist() == expected
    assert set(observed_first).issubset(set(saved.date))
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("kind", [DataDownloader, SmartBatchDownloader])
def test_shared_downloader_keeps_request_identity_and_distinct_paths_parallel(tmp_path, monkeypatch, kind):
    monkeypatch.chdir(tmp_path)
    downloader = kind()
    fetched, publishing = threading.Barrier(2, timeout=2), threading.Barrier(2, timeout=2)
    expected = [("okx", "futures", PAIR, 100.), ("binance", "spot", "BTC/USDT", 200.)]
    def factory(config_id=None, exchange_id=None, trade_type=None):
        price = 100. if exchange_id == "okx" else 200.
        candles = [[row[0], price, price + 1., price - 1., price, 10.] for row in hourly_market()[:24]]
        exchange = client_for(candles)
        exchange.id = exchange_id
        exchange.parse_timeframe.return_value = 3600
        fetch = exchange.fetch_ohlcv.side_effect
        def fetch_together(*args, **kwargs):
            fetched.wait()
            return fetch(*args, **kwargs)
        exchange.fetch_ohlcv.side_effect = fetch_together
        return exchange
    monkeypatch.setattr(downloader, "get_exchange_instance", factory)
    monkeypatch.setattr("core.data_feed.data_downloader.time.sleep", lambda _: None)
    real_write = DataDownloader._write_feather_atomic
    def publish_together(data, path):
        publishing.wait()
        real_write(data, path)
    monkeypatch.setattr(DataDownloader, "_write_feather_atomic", staticmethod(publish_together))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download_hour_day, downloader, 1, exchange_id=venue, trade_type=market, symbol=symbol)
                   for venue, market, symbol, _ in expected]
        outcomes = [future.result(timeout=4) for future in futures]
    assert all(outcome["success"] for outcome in outcomes), outcomes
    for venue, market, symbol, price in expected:
        target = data_path("data", venue, symbol, "1h", market)
        assert target.exists(), f"request identity was lost: {target}"
        saved = pd.read_feather(target)
        assert len(saved) == 24
        assert saved.close.eq(price).all()


@pytest.mark.parametrize("kind", [DataDownloader, SmartBatchDownloader])
@pytest.mark.parametrize("failure", ["write", "read"])
def test_failed_update_releases_target_for_another_downloader(tmp_path, monkeypatch, kind, failure):
    monkeypatch.chdir(tmp_path)
    target = data_path("data", "okx", PAIR, "1h", "futures")
    seed_middle_day(target)
    before = target.read_bytes()
    first = kind()
    second = SmartBatchDownloader() if kind is DataDownloader else DataDownloader()
    for downloader in (first, second):
        exchange = client_for(hourly_market())
        exchange.parse_timeframe.return_value = 3600
        monkeypatch.setattr(downloader, "get_exchange_instance", lambda *args, _exchange=exchange, **kwargs: _exchange)
    monkeypatch.setattr("core.data_feed.data_downloader.time.sleep", lambda _: None)
    with monkeypatch.context() as broken:
        if failure == "write":
            def fail_write(*args):
                raise OSError("bounded publish failure")
            broken.setattr(DataDownloader, "_write_feather_atomic", staticmethod(fail_write))
        else:
            def fail_read(*args, **kwargs):
                raise OSError("bounded read failure")
            broken.setattr(pd, "read_feather", fail_read)
        outcome = download_hour_day(first, 1)
        assert outcome["success"] is False
        assert "bounded" in outcome["error"]
    assert target.read_bytes() == before
    outcomes = []
    # A leaked lock must fail promptly rather than hang the test process.
    worker = threading.Thread(target=lambda: outcomes.append(download_hour_day(second, 1)), daemon=True)
    worker.start()
    worker.join(2)
    assert not worker.is_alive(), "failed transaction retained the target lock"
    assert outcomes[0]["success"], outcomes
    assert len(pd.read_feather(target)) == 48


def run_download(tmp_path, monkeypatch, client, downloader_type=SmartBatchDownloader, days=1):
    monkeypatch.chdir(tmp_path)
    downloader = downloader_type()
    monkeypatch.setattr(downloader, "get_exchange_instance", lambda *args, **kwargs: client)
    monkeypatch.setattr("core.data_feed.data_downloader.time.sleep", lambda _: None)
    progress = Mock()
    arguments = dict(symbol=PAIR, timeframe="1m", start_date="2024-01-01",
                     end_date=f"2024-01-{1 + days:02}", trade_type="futures",
                     progress_callback=progress)
    if downloader_type is SmartBatchDownloader:
        result = downloader.download_ohlcv_batch(exchange_id="okx", **arguments)
    else:
        result = downloader.download_ohlcv(**arguments)
    return result, progress


@pytest.mark.parametrize("downloader_type", [DataDownloader, SmartBatchDownloader])
@pytest.mark.parametrize("page_size", [1000, 137])
def test_one_minute_day_saves_all_1440_bars_with_real_health_gate(tmp_path, monkeypatch, downloader_type, page_size):
    client = client_for(rows(), page_size)
    result, progress = run_download(tmp_path, monkeypatch, client, downloader_type)
    assert result["success"], result
    saved = pd.read_feather(data_path("data", "okx", PAIR, "1m", "futures"))
    count_key = "total_records" if downloader_type is SmartBatchDownloader else "data_points"
    assert len(saved) == result[count_key] == 1440
    assert saved.date.tolist() == list(pd.date_range("2024-01-01", periods=1440, freq="min"))
    assert saved.close.eq(100.).all()
    assert progress.call_args.args[0] == 100
    if page_size == 1000:
        assert [call.kwargs["limit"] for call in client.fetch_ohlcv.call_args_list] == [1000, 440]
        assert [call.args[2] for call in client.fetch_ohlcv.call_args_list] == [START_MS, START_MS + 60000000]


def test_multiple_batches_do_not_skip_each_days_tail(tmp_path, monkeypatch):
    client = client_for(rows(2880))
    result, _ = run_download(tmp_path, monkeypatch, client, days=2)
    assert result["success"], result
    saved = pd.read_feather(data_path("data", "okx", PAIR, "1m", "futures"))
    assert len(saved) == 2880
    assert saved.date.tolist() == list(pd.date_range("2024-01-01", periods=2880, freq="min"))
    assert [call.kwargs["limit"] for call in client.fetch_ohlcv.call_args_list] == [1000, 440, 1000, 440]


@pytest.mark.parametrize("reply", [[], rows(1), rows(1441)[-1:]])
def test_missing_tail_cannot_be_success_even_with_contiguous_prefix(tmp_path, monkeypatch, reply):
    client = client_for([])
    client.fetch_ohlcv.side_effect = [rows(1000), reply, reply, reply]
    result, progress = run_download(tmp_path, monkeypatch, client)
    assert result["success"] is False
    assert "3/3 attempts" in result["error"]
    assert client.fetch_ohlcv.call_count == 4
    assert not data_path("data", "okx", PAIR, "1m", "futures").exists()
    assert all(call.args[0] != 100 for call in progress.call_args_list)


def test_missing_requested_start_fails_with_explicit_denominator(tmp_path, monkeypatch):
    client = client_for(rows()[1:])
    result, _ = run_download(tmp_path, monkeypatch, client)
    assert result["success"] is False
    assert "1439/1440 bars" in result["error"]
    assert not data_path("data", "okx", PAIR, "1m", "futures").exists()


@pytest.mark.parametrize("selection", [slice(1, None), slice(None, -1), slice(1, -1)])
def test_requested_range_detects_missing_edges(selection):
    downloader = DataDownloader()
    data = frame(1440).set_index("date").iloc[selection]
    with pytest.raises(DataGapFillError, match="/1440 bars"):
        downloader._validate_requested_range(data, "1m", "2024-01-01", "2024-01-02", client_for([]))


def test_valid_ohlc_passes_actual_health_and_final_gate():
    data = frame()
    health = health_checker.check_data_health(data, "1m", PAIR)
    assert health["is_healthy"] and health["health_score"] == 100
    assert DataDownloader()._final_validation(data.set_index("date"), "1m", PAIR)


def test_twenty_invalid_ohlc_rows_fail_actual_health_and_final_gate():
    data = frame()
    data["high"], data["low"] = 90., 110.
    health = health_checker.check_data_health(data, "1m", PAIR)
    assert not health["is_healthy"]
    assert health["health_score"] < 100
    assert "20 条记录 (100.00%)" in str(health)
    assert not DataDownloader()._final_validation(data.set_index("date"), "1m", PAIR)


@pytest.mark.parametrize("invalid_count", [1, 1440])
def test_invalid_ohlc_cannot_be_persisted_by_download_entrypoint(tmp_path, monkeypatch, invalid_count):
    candles = rows()
    for candle in candles[:invalid_count]:
        candle[2], candle[3] = 90., 110.
    result, progress = run_download(tmp_path, monkeypatch, client_for(candles))
    assert result["success"] is False
    assert "验证失败" in result["error"]
    assert not data_path("data", "okx", PAIR, "1m", "futures").exists()
    assert all(call.args[0] != 100 for call in progress.call_args_list)


@pytest.mark.parametrize("downloader_type", [DataDownloader, SmartBatchDownloader])
@pytest.mark.parametrize("failure_stage", ["write", "replace"])
@pytest.mark.parametrize("existing", [False, True])
def test_write_failure_is_reported_and_preserves_existing_file(
        tmp_path, monkeypatch, existing, failure_stage, downloader_type):
    monkeypatch.chdir(tmp_path)
    target = data_path("data", "okx", PAIR, "1m", "futures")
    target.parent.mkdir(parents=True)
    if existing:
        frame().to_feather(target)
    original = target.read_bytes() if existing else None

    def partial_write_then_fail(self, path, *args, **kwargs):
        Path(path).write_bytes(b"partial feather data")
        raise OSError("disk full test")

    if failure_stage == "write":
        monkeypatch.setattr(pd.DataFrame, "to_feather", partial_write_then_fail)
    else:
        def fail_replace(self, target):
            raise OSError("disk full test during replace")
        monkeypatch.setattr(Path, "replace", fail_replace)
    result, progress = run_download(tmp_path, monkeypatch, client_for(rows()), downloader_type)
    assert result["success"] is False
    assert "disk full test" in result["error"]
    assert all(call.args[0] != 100 for call in progress.call_args_list)
    if existing:
        assert target.read_bytes() == original
    else:
        assert not target.exists()
    assert list(target.parent.iterdir()) == ([target] if existing else [])


def old_market_frame():
    old = frame(20)
    old["date"] -= pd.Timedelta(minutes=20)
    old["date"] = old.date.astype("datetime64[ms]")
    old["close"] += np.arange(len(old)) * .01
    old["open"] = old.close - .005
    old["high"], old["low"] = old.close + 1., old.close - 1.
    return old


def old_period_reader():
    return RealDataClient({"target": {"entry_price": "next_open", "horizon_bars": 2}, "data_feeds": {
        "exchange": "okx", "instrument_type": "futures", "pairs": [PAIR], "timeframe": "1m",
        "mine_period": [["2023-12-31 23:40", "2023-12-31 23:59"]],
    }})


@pytest.mark.parametrize("downloader_type", [DataDownloader, SmartBatchDownloader])
@pytest.mark.parametrize("time_column", ["date", "time", "timestamp"])
def test_merge_preserves_reader_supported_time_columns_and_labels(
        tmp_path, monkeypatch, downloader_type, time_column):
    monkeypatch.chdir(tmp_path)
    target = data_path("data", "okx", PAIR, "1m", "futures")
    target.parent.mkdir(parents=True)
    old = old_market_frame()
    expected = old.set_index("date")
    if time_column == "timestamp":
        old["date"] = old.date.to_numpy(dtype="datetime64[ms]").astype("int64")
    else:
        old["date"] = old.date.dt.tz_localize("UTC").dt.tz_convert("Asia/Shanghai")
    old.rename(columns={"date": time_column}).to_feather(target)
    before = old_period_reader()
    pd.testing.assert_frame_equal(before.get_data().rename_axis("date"), expected)

    result, _ = run_download(tmp_path, monkeypatch, client_for(rows()), downloader_type)
    assert result["success"], result
    saved = pd.read_feather(target)
    assert len(saved) == 1460
    assert saved.date.notna().all()
    assert saved.date.dt.tz is None
    assert saved.columns.tolist() == ["date", "open", "high", "low", "close", "volume"]
    after = old_period_reader()
    pd.testing.assert_frame_equal(after.get_data(), expected)
    expected_returns = pd.Series(np.nan, index=expected.index, name="returns")
    expected_returns.iloc[:-2] = expected.close.to_numpy()[2:] / expected.open.to_numpy()[1:-1] - 1
    pd.testing.assert_series_equal(after.get_returns(), expected_returns)
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("downloader_type", [DataDownloader, SmartBatchDownloader])
@pytest.mark.parametrize("old_format", ["corrupt", "unknown", "invalid_time", "missing_time"])
def test_unreadable_existing_time_data_fails_closed_without_alternate_success(
        tmp_path, monkeypatch, downloader_type, old_format):
    monkeypatch.chdir(tmp_path)
    target = data_path("data", "okx", PAIR, "1m", "futures")
    target.parent.mkdir(parents=True)
    old = old_market_frame()
    if old_format == "corrupt":
        target.write_bytes(b"corrupt feather fixture")
    else:
        if old_format == "unknown":
            old = old.rename(columns={"date": "unknown_clock"})
        elif old_format == "invalid_time":
            old["date"] = ["not a timestamp"] * len(old)
        else:
            old.loc[0, "date"] = pd.NaT
        old.to_feather(target)
    original = target.read_bytes()
    result, progress = run_download(tmp_path, monkeypatch, client_for(rows()), downloader_type)
    assert not result["success"], result
    assert target.read_bytes() == original
    assert list(target.parent.iterdir()) == [target]
    assert all(call.args[0] != 100 for call in progress.call_args_list)


@pytest.mark.parametrize("downloader_type", [DataDownloader, SmartBatchDownloader])
def test_legacy_time_rows_are_included_in_final_health_check(tmp_path, monkeypatch, downloader_type):
    monkeypatch.chdir(tmp_path)
    target = data_path("data", "okx", PAIR, "1m", "futures")
    target.parent.mkdir(parents=True)
    old = old_market_frame()
    old["date"] = old.date.to_numpy(dtype="datetime64[ms]").astype("int64")
    old["high"], old["low"] = 90., 110.
    old.rename(columns={"date": "timestamp"}).to_feather(target)
    original = target.read_bytes()
    result, _ = run_download(tmp_path, monkeypatch, client_for(rows()), downloader_type)
    assert not result["success"], result
    assert target.read_bytes() == original
    assert list(target.parent.iterdir()) == [target]


@pytest.mark.parametrize("column,value", [
    ("open", np.nan), ("high", np.inf), ("low", -np.inf), ("close", np.nan), ("volume", np.inf),
])
def test_non_finite_required_values_fail_real_health_even_below_missing_ratio_threshold(column, value):
    data = frame(1440)
    data.loc[0, column] = value
    health = health_checker.check_data_health(data, "1m", PAIR)
    assert not health["is_healthy"]
    assert f"列 {column} 包含非有限数值: 1 条记录" in health["issues"]
    assert not DataDownloader()._final_validation(data.set_index("date"), "1m", PAIR)


def test_nullable_missing_numeric_value_is_not_treated_as_finite():
    data = frame(1440)
    data["volume"] = data.volume.astype("Float64")
    data.loc[0, "volume"] = pd.NA
    health = health_checker.check_data_health(data, "1m", PAIR)
    assert not health["is_healthy"]
    assert "列 volume 包含非有限数值: 1 条记录" in health["issues"]


def run_public_cli(monkeypatch, client, mode="overwrite"):
    monkeypatch.setattr("core.data_feed.data_downloader.ccxt.okx", Mock(return_value=client))
    monkeypatch.setattr("core.data_feed.data_downloader.time.sleep", lambda _: None)
    monkeypatch.setattr(sys, "argv", ["factorminer", "download", "--exchange", "okx", "--symbols", PAIR,
        "--timeframes", "1m", "--type", "futures", "--start", "2024-01-01", "--end", "2024-01-02", "--mode", mode])
    return cli.main()


def test_public_cli_overwrite_ignores_old_interval_and_old_health_errors(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = data_path("data", "okx", PAIR, "1m", "futures")
    target.parent.mkdir(parents=True)
    old = frame(1440)
    old["date"] -= pd.Timedelta(days=1)
    old["high"], old["low"] = 90., 110.
    old.to_feather(target)
    assert run_public_cli(monkeypatch, client_for(rows())) is None
    saved = pd.read_feather(target)
    assert len(saved) == 1440
    assert saved.date.tolist() == frame(1440).date.tolist()
    assert saved.high.eq(101.).all() and saved.low.eq(99.).all()


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("column,value", [("high", np.inf), ("volume", np.nan)])
def test_public_cli_non_finite_download_fails_without_publishing(tmp_path, monkeypatch, existing, column, value):
    monkeypatch.chdir(tmp_path)
    target = data_path("data", "okx", PAIR, "1m", "futures")
    target.parent.mkdir(parents=True)
    if existing:
        frame().to_feather(target)
    original = target.read_bytes() if existing else None
    candles = rows()
    candles[0][["date", "open", "high", "low", "close", "volume"].index(column)] = value
    with pytest.raises(SystemExit) as failure:
        run_public_cli(monkeypatch, client_for(candles))
    assert failure.value.code == 1
    if existing:
        assert target.read_bytes() == original
    else:
        assert not target.exists()
    assert list(target.parent.iterdir()) == ([target] if existing else [])


def test_public_cli_incomplete_overwrite_preserves_old_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = data_path("data", "okx", PAIR, "1m", "futures")
    target.parent.mkdir(parents=True)
    frame().to_feather(target)
    original = target.read_bytes()
    with pytest.raises(SystemExit) as failure:
        run_public_cli(monkeypatch, client_for(rows(1000)))
    assert failure.value.code == 1
    assert target.read_bytes() == original
    assert list(target.parent.iterdir()) == [target]
