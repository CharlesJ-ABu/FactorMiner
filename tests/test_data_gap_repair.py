"""Offline regression tests for exchange identity, UTC and bounded gap repair."""

import os
import stat
from unittest.mock import Mock

import pandas as pd
import pytest

from config.user_config import UserConfigManager
from core.data_feed.batch_downloader import SmartBatchDownloader
from core.data_feed.data_downloader import DataDownloader, DataGapFillError
from core.data_feed.naming import data_path
from core.data_feed.processor import from_epoch_milliseconds
from core.data_feed.real_client import RealDataClient


PAIR = "BTC/USDT:USDT"
START_MS = int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000)


def test_exchange_preferences_stay_in_the_active_workspace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    manager = UserConfigManager()
    assert manager.config_file == tmp_path / "config" / "user_config" / "config.json"
    assert not manager.config_file.exists()
    assert manager.set_preference("exchange", "okx")
    assert UserConfigManager().get_preference("exchange") == "okx"
    if os.name == "posix":
        assert stat.S_IMODE(manager.config_file.stat().st_mode) == 0o600


def test_config_write_failure_preserves_previous_preferences(tmp_path):
    manager = UserConfigManager(config_dir=tmp_path)
    assert manager.set_preference("exchange", "okx")
    manager.config["preferences"]["invalid"] = object()
    with pytest.raises(TypeError):
        manager.save_config()
    assert UserConfigManager(config_dir=tmp_path).config["preferences"] == {"exchange": "okx"}
    assert list(tmp_path.iterdir()) == [manager.config_file]


@pytest.mark.parametrize("downloader", [DataDownloader, SmartBatchDownloader])
def test_price_repair_forward_fills_from_existing_prices(downloader):
    frame = pd.DataFrame({"close": [100.0, 0.0, float("inf"), 105.0]})
    repaired = downloader()._fix_price_issues(frame)
    assert repaired["close"].tolist() == [100.0, 100.0, 100.0, 105.0]


def test_fractional_epoch_milliseconds_keep_submillisecond_precision():
    dates = from_epoch_milliseconds(pd.Series([1000.5]))
    assert dates.iloc[0] == pd.Timestamp("1970-01-01 00:00:01.000500")


def candle(timestamp, close=100.0):
    return [timestamp, close, close + 1, close - 1, close, 10.0]


def frame(times=None):
    times = times if times is not None else pd.to_datetime([START_MS, START_MS + 120000], unit="ms")
    return pd.DataFrame({"open": [100., 102.], "high": [101., 103.], "low": [99., 101.],
                         "close": [100., 102.], "volume": [10., 10.]},
                        index=pd.DatetimeIndex(times, name="date"))


def exchange(responses, seconds=60):
    result = Mock()
    result.id = "okx"
    result.rateLimit = 10
    result.parse_timeframe.return_value = seconds
    result.fetch_ohlcv.side_effect = responses
    return result


@pytest.mark.parametrize("tz", [None, "UTC", "Asia/Shanghai", "America/New_York"])
def test_detect_download_merge_preserves_utc_instants(tz, monkeypatch):
    downloader = DataDownloader()
    times = pd.to_datetime([START_MS, START_MS + 120000], unit="ms", utc=True)
    times = times.tz_convert(tz) if tz else times.tz_localize(None)
    original = frame(times)
    before = original.copy(deep=True)
    gaps = downloader._detect_data_gaps(original, "1m")
    assert len(gaps) == 1
    assert gaps[0]["start_time"] == pd.Timestamp("2024-01-01 00:01")
    assert gaps[0]["missing_intervals"] == 1
    client = exchange([[candle(START_MS + 60000, 101.)]])
    monkeypatch.setattr(downloader, "get_exchange_instance", Mock(side_effect=AssertionError("must reuse exchange")))
    actual = downloader._download_missing_data(original, gaps, PAIR, "1m", exchange=client)
    client.fetch_ohlcv.assert_called_once_with(PAIR, "1m", START_MS + 60000, limit=1)
    assert actual.index.tolist() == list(pd.date_range("2024-01-01", periods=3, freq="min"))
    assert actual.close.tolist() == [100., 101., 102.]
    assert not actual.index.duplicated().any()
    pd.testing.assert_frame_equal(original, before)


def test_aware_gap_bounds_preserve_the_requested_instant():
    downloader = DataDownloader()
    original = frame(pd.DatetimeIndex(["2024-01-01 08:00", "2024-01-01 08:02"], tz="Asia/Shanghai"))
    gaps = [{"start_time": pd.Timestamp("2024-01-01 08:01", tz="Asia/Shanghai"),
             "end_time": pd.Timestamp("2024-01-01 08:02", tz="Asia/Shanghai")}]
    client = exchange([[candle(START_MS + 60000, 101.)]])
    downloader._download_missing_data(original, gaps, PAIR, "1m", exchange=client)
    assert client.fetch_ohlcv.call_args.args[2] == START_MS + 60000


@pytest.mark.parametrize("response", [[], [candle(START_MS)], [candle(START_MS + 600000)]])
def test_empty_stale_and_out_of_gap_responses_stop_after_three_attempts(response, monkeypatch):
    downloader = DataDownloader()
    original = frame()
    client = exchange([response] * 3)
    sleep = Mock()
    monkeypatch.setattr("core.data_feed.data_downloader.time.sleep", sleep)
    callback = Mock()
    with pytest.raises(DataGapFillError, match="Unresolved gap.*3/3 attempts"):
        downloader._fill_data_gaps(original, PAIR, "1m", "2024-01-01", "2024-01-02",
                                   callback, exchange=client)
    assert client.fetch_ohlcv.call_count == 3
    assert all(call.args[2] == START_MS + 60000 for call in client.fetch_ohlcv.call_args_list)
    assert sleep.call_count == 2
    assert "Unresolved gap" in callback.call_args.args[1]
    pd.testing.assert_frame_equal(original, frame())


def test_transient_empty_response_recovers_without_overwriting_existing_bars(monkeypatch):
    downloader = DataDownloader()
    original = frame()
    client = exchange([[], [candle(START_MS, 999.), candle(START_MS + 60000, 101.),
                            candle(START_MS + 120000, 999.)]])
    monkeypatch.setattr("core.data_feed.data_downloader.time.sleep", lambda _: None)
    actual = downloader._fill_data_gaps(original, PAIR, "1m", "2024-01-01", "2024-01-02", exchange=client)
    assert actual.close.tolist() == [100., 101., 102.]
    assert client.fetch_ohlcv.call_count == 2


def test_gap_repair_without_original_exchange_fails_closed():
    downloader = DataDownloader()
    original = frame()
    with pytest.raises(DataGapFillError, match="original download exchange"):
        downloader._download_missing_data(original, downloader._detect_data_gaps(original, "1m"), PAIR, "1m")


def test_merge_reads_only_the_selected_exchange_and_normalizes_time(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    downloader = SmartBatchDownloader()
    downloader.exchange_id, downloader.trade_type = "okx", "futures"
    for venue, value in [("okx", 100.), ("binance", 999.)]:
        target = data_path("data", venue, PAIR, "1m", "futures")
        target.parent.mkdir(parents=True, exist_ok=True)
        old = frame().iloc[:1].reset_index()
        old["close"] = value
        old.to_feather(target)
    new = frame().iloc[1:]
    new.index = new.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
    merged = downloader._merge_with_existing_data(new, PAIR, "1m", "2024-01-01", "2024-01-02")
    assert merged.close.tolist() == [100., 102.]
    assert merged.index.tz is None
    assert merged.index.tolist() == frame().index.tolist()


@pytest.mark.parametrize("downloader_type", [DataDownloader, SmartBatchDownloader])
@pytest.mark.parametrize("existing_aware_file", [False, True])
def test_download_reuses_exchange_and_config_for_gap_repair(tmp_path, monkeypatch, downloader_type, existing_aware_file):
    monkeypatch.chdir(tmp_path)
    if existing_aware_file:
        target = data_path("data", "okx", PAIR, "1m", "futures")
        target.parent.mkdir(parents=True)
        old = frame()
        old.index = old.index.tz_localize("UTC").tz_convert("Asia/Shanghai")
        old.reset_index().to_feather(target)
    downloader = downloader_type()
    # Exercise the full requested day, including pagination and an internal gap.
    initial = [candle(START_MS + i * 60000, 100. + i * .01) for i in range(1001) if i != 1]
    tail = [candle(START_MS + i * 60000, 100. + i * .01) for i in range(1001, 1440)]
    client = exchange([initial, tail, [candle(START_MS + 60000, 100.01)]])
    factory = Mock(return_value=client)
    monkeypatch.setattr(downloader, "get_exchange_instance", factory)
    monkeypatch.setattr("core.data_feed.data_downloader.time.sleep", lambda _: None)
    kwargs = dict(config_id=17, symbol=PAIR, timeframe="1m", start_date="2024-01-01",
                  end_date="2024-01-02", trade_type="futures")
    result = (downloader.download_ohlcv_batch(**kwargs, exchange_id="okx")
              if isinstance(downloader, SmartBatchDownloader) else downloader.download_ohlcv(**kwargs))
    assert result["success"], result
    assert factory.call_count == 1
    assert factory.call_args.args[0] == 17
    assert factory.call_args.kwargs["trade_type"] == "futures"
    assert client.fetch_ohlcv.call_count == 3
    assert client.fetch_ohlcv.call_args_list[0].args[2] == START_MS
    assert client.fetch_ohlcv.call_args_list[1].args[2] == START_MS + 1001 * 60000
    assert client.fetch_ohlcv.call_args_list[2].args[2] == START_MS + 60000
    saved = pd.read_feather(data_path("data", "okx", PAIR, "1m", "futures"))
    assert saved.close.tolist() == [100. + i * .01 for i in range(1440)]
    assert pd.to_datetime(saved.date, utc=True).tolist() == list(pd.date_range("2024-01-01", periods=1440, freq="min", tz="UTC"))
    assert not data_path("data", "binance", PAIR, "1m", "futures").exists()


def test_failed_gap_download_does_not_overwrite_existing_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = data_path("data", "okx", PAIR, "1m", "futures")
    target.parent.mkdir(parents=True)
    frame().reset_index().to_feather(target)
    before = target.read_bytes()
    downloader = SmartBatchDownloader()
    client = exchange([[candle(START_MS), candle(START_MS + 120000)], [], [], []])
    monkeypatch.setattr(downloader, "get_exchange_instance", lambda *args, **kwargs: client)
    monkeypatch.setattr("core.data_feed.data_downloader.time.sleep", lambda _: None)
    save = Mock(side_effect=AssertionError("incomplete data must not be saved"))
    monkeypatch.setattr(downloader, "_save_data_with_merge", save)
    result = downloader.download_ohlcv_batch(exchange_id="okx", symbol=PAIR, timeframe="1m",
        start_date="2024-01-01", end_date="2024-01-02", trade_type="futures")
    assert not result["success"]
    assert "Unresolved gap" in result["error"]
    assert client.fetch_ohlcv.call_count == 4
    save.assert_not_called()
    assert target.read_bytes() == before


def test_exchange_config_honors_the_selected_market(monkeypatch):
    factory = Mock()
    monkeypatch.setattr("core.data_feed.data_downloader.ccxt.okx", factory)
    monkeypatch.setattr("core.data_feed.data_downloader.config_manager.get_exchange_config",
        lambda _: dict(exchange_id="okx", api_key="", secret="", password="", sandbox=True))
    DataDownloader().get_exchange_instance(config_id=17, trade_type="futures")
    assert factory.call_args.args[0]["options"]["defaultType"] == "future"


@pytest.mark.parametrize("venue,timeframe,bounds,expected_first,expected_last", [
    ("okx", "1m", ["2024-01-01 08:00+08:00", "2024-01-01 08:19+08:00"],
     "2024-01-01 00:00", "2024-01-01 00:19"),
    ("binance", "1m", ["2024-01-01 00:00:30", "2024-01-01 00:19:30"],
     "2024-01-01 00:01", "2024-01-01 00:19"),
    ("okx", "1m", ["2023-12-31 19:00:30-05:00", "2023-12-31 19:19:30-05:00"],
     "2024-01-01 00:01", "2024-01-01 00:19"),
    ("binance", "1h", ["2024-01-01 00:00", "2024-01-02 00:00"],
     "2024-01-01 00:00", "2024-01-02 00:00"),
    ("okx", "1h", ["2024-01-01 08:00:30+08:00", "2024-01-02 08:00:30+08:00"],
     "2024-01-01 01:00", "2024-01-02 00:00"),
    # The fixture exchange has no data after 10:19; requesting a full extra day must fail.
    ("okx", "1m", ["2024-01-01 10:00:00Z", "2024-01-01 10:19:40Z"],
     "2024-01-01 10:00", "2024-01-01 10:19"),
])
def test_real_client_auto_download_preserves_identity_inclusive_bars_and_labels(
        tmp_path, monkeypatch, venue, timeframe, bounds, expected_first, expected_last):
    monkeypatch.chdir(tmp_path)
    expected_index = pd.date_range(expected_first, expected_last, freq="min" if timeframe == "1m" else "h", name="date")
    candles = [candle(int(timestamp.timestamp() * 1000), 100. + i * .01)
               for i, timestamp in enumerate(expected_index.tz_localize("UTC"))]
    client = exchange([], seconds=60 if timeframe == "1m" else 3600)
    client.id = venue
    client.rateLimit = 0
    client.fetch_ohlcv.side_effect = lambda symbol, timeframe, since, limit: [
        row for row in candles if row[0] >= since
    ][:limit]
    selected = Mock(return_value=client)
    other = Mock(side_effect=AssertionError("configured exchange must not be replaced"))
    monkeypatch.setattr(f"core.data_feed.data_downloader.ccxt.{venue}", selected)
    monkeypatch.setattr(f"core.data_feed.data_downloader.ccxt.{'binance' if venue == 'okx' else 'okx'}", other)
    monkeypatch.setattr("core.data_feed.data_downloader.time.sleep", lambda _: None)

    reader = RealDataClient({"target": {"entry_price": "next_open", "horizon_bars": 2}, "data_feeds": {
        "exchange": venue, "instrument_type": "futures", "timeframe": timeframe,
        "pairs": [PAIR], "mine_period": [bounds], "test_period": [],
    }})
    data = reader.get_data()
    assert data.index.equals(expected_index)
    assert data.close.tolist() == [100. + i * .01 for i in range(len(expected_index))]
    expected_returns = data.close.shift(-2).div(data.open.shift(-1)).sub(1.).rename("returns")
    pd.testing.assert_series_equal(reader.get_returns(), expected_returns)
    selected.assert_called_once()
    assert selected.call_args.args[0]["options"]["defaultType"] == "future"
    other.assert_not_called()
    client.fetch_ohlcv.assert_called_once_with(PAIR, timeframe, candles[0][0], limit=len(candles))
    target = data_path("data", venue, PAIR, timeframe, "futures")
    assert pd.read_feather(target).date.tolist() == expected_index.tolist()
    assert len(list(tmp_path.rglob("*.feather"))) == 1


def test_timestamp_download_uses_half_open_bar_times_without_requesting_later_data(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    candles = [candle(START_MS + i * 60000, 100. + i * .01) for i in range(1, 21)]
    client = exchange([])
    client.rateLimit = 0
    client.fetch_ohlcv.side_effect = lambda symbol, timeframe, since, limit: [
        row for row in candles if row[0] >= since
    ][:limit]
    monkeypatch.setattr("core.data_feed.data_downloader.ccxt.okx", Mock(return_value=client))
    result = DataDownloader().download_ohlcv(exchange_id="okx", symbol=PAIR, timeframe="1m",
        start_date="2024-01-01 08:00:30+08:00", end_date="2024-01-01 08:20:30+08:00", trade_type="futures")
    assert result["success"], result
    saved = pd.read_feather(data_path("data", "okx", PAIR, "1m", "futures"))
    assert saved.date.tolist() == list(pd.date_range("2024-01-01 00:01", periods=20, freq="min"))
    client.fetch_ohlcv.assert_called_once_with(PAIR, "1m", START_MS + 60000, limit=20)
