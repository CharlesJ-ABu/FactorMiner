import unittest
from pathlib import Path

from core.data_feed.naming import data_filename, data_path, parse_data_filename


class DataNamingTests(unittest.TestCase):
    def test_futures_filename_uses_ccxt_settlement_suffix(self):
        self.assertEqual(
            data_filename("BTC/USDT:USDT", "1m", "futures"),
            "BTC_USDT_USDT-1m-futures.feather",
        )

    def test_spot_filename(self):
        self.assertEqual(
            data_filename("BTC/USDT", "1m", "spot"),
            "BTC_USDT-1m-spot.feather",
        )

    def test_path_is_canonical(self):
        self.assertEqual(
            data_path("data", "binance", "BTC/USDT:USDT", "1m", "futures"),
            Path("data/binance/futures/BTC_USDT_USDT-1m-futures.feather"),
        )

    def test_parse_canonical_filename(self):
        self.assertEqual(
            parse_data_filename("BTC_USDT_USDT-1m-futures.feather"),
            ("BTC_USDT_USDT", "1m", "futures"),
        )

    def test_futures_without_settlement_suffix_is_rejected(self):
        with self.assertRaises(ValueError):
            data_filename("BTC/USDT", "1m", "futures")
        with self.assertRaises(ValueError):
            parse_data_filename("BTC_USDT-1m-futures.feather")


if __name__ == "__main__":
    unittest.main()
