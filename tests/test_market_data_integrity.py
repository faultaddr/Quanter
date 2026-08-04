"""Integrity and provenance tests for historical market data."""

from datetime import datetime, timezone
import unittest
from unittest.mock import patch

import pandas as pd

from quanttool.core.errors import DataNotAvailableError, ValidationError


def make_daily_frame() -> pd.DataFrame:
    """Build deterministic daily bars for validation tests."""
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-08-03", "2026-08-04"]),
            "open": [10.0, 10.2],
            "high": [10.5, 10.6],
            "low": [9.9, 10.1],
            "close": [10.3, 10.4],
            "volume": [1000.0, 1200.0],
            "amount": [10300.0, 12480.0],
        }
    )


class MarketDataValidationTests(unittest.TestCase):
    """Reject malformed or unattributed market data."""

    def test_validation_attaches_concrete_provenance(self):
        from quanttool.infrastructure.data_providers.validation import (
            DataProvenance,
            validate_market_data,
        )

        result = validate_market_data(
            make_daily_frame(),
            start_date=datetime(2026, 8, 3),
            end_date=datetime(2026, 8, 4),
            provenance=DataProvenance(
                provider="tencent",
                retrieved_at=datetime(2026, 8, 4, tzinfo=timezone.utc),
                frequency="1d",
                adjustment="qfq",
                simulated=False,
            ),
        )
        self.assertEqual(
            result.attrs["quanttool_provenance"]["provider"],
            "tencent",
        )
        self.assertFalse(result.attrs["quanttool_provenance"]["simulated"])

    def test_validation_rejects_duplicate_timestamp(self):
        from quanttool.infrastructure.data_providers.validation import (
            DataProvenance,
            validate_market_data,
        )

        frame = make_daily_frame()
        frame.loc[1, "timestamp"] = frame.loc[0, "timestamp"]
        with self.assertRaises(ValidationError):
            validate_market_data(
                frame,
                datetime(2026, 8, 3),
                datetime(2026, 8, 4),
                DataProvenance(
                    "sina",
                    datetime.now(timezone.utc),
                    "1d",
                    "qfq",
                    False,
                ),
            )

    def test_validation_rejects_bad_ohlc_and_negative_volume(self):
        from quanttool.infrastructure.data_providers.validation import (
            DataProvenance,
            validate_market_data,
        )

        provenance = DataProvenance(
            "sina",
            datetime.now(timezone.utc),
            "1d",
            "qfq",
            False,
        )
        bad_ohlc = make_daily_frame()
        bad_ohlc.loc[0, "high"] = 9.0
        with self.assertRaises(ValidationError):
            validate_market_data(
                bad_ohlc,
                datetime(2026, 8, 3),
                datetime(2026, 8, 4),
                provenance,
            )
        negative_volume = make_daily_frame()
        negative_volume.loc[0, "volume"] = -1
        with self.assertRaises(ValidationError):
            validate_market_data(
                negative_volume,
                datetime(2026, 8, 3),
                datetime(2026, 8, 4),
                provenance,
            )

    def test_validation_rejects_descending_out_of_range_and_non_numeric_data(self):
        from quanttool.infrastructure.data_providers.validation import (
            DataProvenance,
            validate_market_data,
        )

        provenance = DataProvenance(
            "sina",
            datetime.now(timezone.utc),
            "1d",
            "qfq",
            False,
        )
        descending = make_daily_frame().iloc[::-1].reset_index(drop=True)
        out_of_range = make_daily_frame()
        out_of_range.loc[0, "timestamp"] = pd.Timestamp("2026-08-02")
        non_numeric = make_daily_frame()
        non_numeric["close"] = non_numeric["close"].astype(object)
        non_numeric.loc[0, "close"] = "not-a-price"
        negative_amount = make_daily_frame()
        negative_amount.loc[0, "amount"] = -1

        for frame in [descending, out_of_range, non_numeric, negative_amount]:
            with self.subTest(frame=frame):
                with self.assertRaises(ValidationError):
                    validate_market_data(
                        frame,
                        datetime(2026, 8, 3),
                        datetime(2026, 8, 4),
                        provenance,
                    )

    def test_ashare_fallback_records_tencent(self):
        from quanttool.infrastructure.data_providers.historical.enhanced_fetcher import (
            AshareFetcher,
        )

        with patch.object(
            AshareFetcher,
            "_get_price_sina",
            return_value=pd.DataFrame(),
        ), patch.object(
            AshareFetcher,
            "_get_price_day_tx",
            return_value=make_daily_frame(),
        ):
            result = AshareFetcher.get_price(
                "600000.SH",
                count=2,
                frequency="1d",
            )
        self.assertEqual(result.attrs["concrete_source"], "tencent")

    def test_ashare_provider_never_manufactures_missing_symbol(self):
        from quanttool.infrastructure.data_providers.historical.ashare_provider import (
            AShareProvider,
        )

        class EmptyFetcher:
            @classmethod
            def get_price(cls, *args, **kwargs):
                return pd.DataFrame()

        provider = AShareProvider(fetcher=EmptyFetcher, max_missing_ratio=0.0)
        with self.assertRaises(DataNotAvailableError):
            provider.get_bars(
                ["600000.SH"],
                datetime(2026, 8, 3),
                datetime(2026, 8, 4),
                "1d",
            )


if __name__ == "__main__":
    unittest.main()
