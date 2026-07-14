import unittest

import pandas as pd

from tests.fixtures.algorithm_data import (
    make_breakout_ohlcv,
    make_indicator_ready_ohlcv,
    make_sideways_ohlcv,
    make_trending_ohlcv,
)


class AlgorithmFixtureTests(unittest.TestCase):
    def test_ohlcv_fixtures_have_required_columns(self):
        required = {
            "timestamp",
            "date",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "amount",
        }

        for builder in [
            make_trending_ohlcv,
            make_sideways_ohlcv,
            make_breakout_ohlcv,
        ]:
            with self.subTest(builder=builder.__name__):
                df = builder(rows=260)
                self.assertEqual(len(df), 260)
                self.assertTrue(required.issubset(df.columns))
                self.assertFalse(df[list(required - {"date"})].isna().any().any())
                self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))

    def test_fixtures_are_deterministic(self):
        first = make_trending_ohlcv(rows=260)
        second = make_trending_ohlcv(rows=260)
        pd.testing.assert_frame_equal(first, second)

    def test_indicator_ready_fixture_contains_legacy_columns(self):
        df = make_indicator_ready_ohlcv(rows=260)
        for column in [
            "ma_5",
            "ma_10",
            "ma_20",
            "ma_50",
            "ma_200",
            "atr_14",
            "boll_upper",
            "boll_mid",
            "boll_lower",
            "rsi_24",
            "wr",
            "cci",
        ]:
            self.assertIn(column, df.columns)
        self.assertGreater(df["ma_20"].iloc[-1], 0)
        self.assertGreater(df["atr_14"].iloc[-1], 0)


if __name__ == "__main__":
    unittest.main()
