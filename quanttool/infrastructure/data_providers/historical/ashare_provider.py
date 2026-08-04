"""Attributed adapter for the built-in Sina/Tencent Ashare fetcher."""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Type

import pandas as pd

from ....core.errors import (
    DataNotAvailableError,
    UnsupportedOperationError,
    ValidationError,
)
from ....core.logging import get_logger
from ....core.registry import ComponentType, registry
from ....core.runtime import RuntimeMode, get_runtime_mode
from ....domain.interfaces.data_provider import IDataProvider
from ..validation import (
    DataProvenance,
    validate_batch_completeness,
    validate_market_data,
)
from .enhanced_fetcher import AshareFetcher


logger = get_logger(__name__)


@registry.register(ComponentType.DATA_PROVIDER, "ashare")
class AShareProvider(IDataProvider):
    """Serve only real, validated and attributable Ashare market data."""

    _SUPPORTED_TIMEFRAMES = {"1d", "1w", "1M"}

    def __init__(
        self,
        fetcher: Type[AshareFetcher] = AshareFetcher,
        max_missing_ratio: Optional[float] = None,
    ) -> None:
        """Configure the real fetcher and missing-symbol threshold."""
        self.fetcher = fetcher
        if max_missing_ratio is None:
            mode = get_runtime_mode()
            self.max_missing_ratio = (
                0.0 if mode is RuntimeMode.PRODUCTION else 0.05
            )
        else:
            self.max_missing_ratio = max_missing_ratio
        if not 0.0 <= self.max_missing_ratio <= 1.0:
            raise ValidationError("max_missing_ratio must be between 0 and 1")
        self._initialized = False

    def initialize(self) -> None:
        """Mark the stateless built-in fetcher ready for use."""
        self._initialized = True

    def get_supported_symbols(self) -> List[str]:
        """Reject synthetic symbol universes."""
        raise UnsupportedOperationError(
            "Ashare symbol-universe endpoint is not implemented"
        )

    def get_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Fetch, constrain, validate and attribute historical bars."""
        if timeframe not in self._SUPPORTED_TIMEFRAMES:
            raise UnsupportedOperationError(
                f"Ashare provider does not support timeframe {timeframe!r}"
            )
        if start_date > end_date:
            raise ValidationError("start_date must not be after end_date")
        if not self._initialized:
            self.initialize()

        results: Dict[str, pd.DataFrame] = {}
        count = max(1, (end_date.date() - start_date.date()).days + 1)
        end_value = end_date.strftime("%Y-%m-%d")

        for symbol in symbols:
            try:
                raw = self.fetcher.get_price(
                    symbol,
                    end_date=end_value,
                    count=count,
                    frequency=timeframe,
                )
            except Exception as exc:
                logger.warning("Ashare fetch failed for %s: %s", symbol, exc)
                continue

            if raw is None or raw.empty:
                continue
            source = raw.attrs.get("concrete_source")
            if not source:
                raise ValidationError(
                    f"Ashare data for {symbol} has no concrete source"
                )

            frame = raw.copy()
            try:
                frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValidationError(
                    f"Ashare data for {symbol} has invalid timestamps"
                ) from exc

            mask = (frame["timestamp"] >= pd.Timestamp(start_date)) & (
                frame["timestamp"] <= pd.Timestamp(end_date)
            )
            frame = frame.loc[mask].copy()
            if frame.empty:
                continue
            frame["timeframe"] = timeframe
            frame["symbol"] = symbol

            results[symbol] = validate_market_data(
                frame,
                start_date=start_date,
                end_date=end_date,
                provenance=DataProvenance(
                    provider=str(source),
                    retrieved_at=datetime.now(timezone.utc),
                    frequency=timeframe,
                    adjustment="qfq",
                    simulated=False,
                ),
            )

        validate_batch_completeness(
            requested=symbols,
            results=results,
            max_missing_ratio=self.max_missing_ratio,
        )
        return results

    def get_latest_bar(
        self,
        symbol: str,
        timeframe: str = "10m",
    ) -> Optional[pd.DataFrame]:
        """Return the latest validated real bar for a supported timeframe."""
        if timeframe not in self._SUPPORTED_TIMEFRAMES:
            raise UnsupportedOperationError(
                f"Ashare provider does not support timeframe {timeframe!r}"
            )
        raw = self.fetcher.get_price(
            symbol,
            end_date=datetime.now().strftime("%Y-%m-%d"),
            count=1,
            frequency=timeframe,
        )
        if raw is None or raw.empty:
            raise DataNotAvailableError(
                f"No latest Ashare bar available for {symbol}"
            )
        source = raw.attrs.get("concrete_source")
        if not source:
            raise ValidationError(
                f"Ashare data for {symbol} has no concrete source"
            )

        frame = raw.tail(1).copy()
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame["timeframe"] = timeframe
        frame["symbol"] = symbol
        timestamp = frame["timestamp"].iloc[0].to_pydatetime()
        return validate_market_data(
            frame,
            start_date=timestamp,
            end_date=timestamp,
            provenance=DataProvenance(
                provider=str(source),
                retrieved_at=datetime.now(timezone.utc),
                frequency=timeframe,
                adjustment="qfq",
                simulated=False,
            ),
        )

    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """Reject hard-coded search results."""
        raise UnsupportedOperationError(
            "Ashare symbol-search endpoint is not implemented"
        )

    def get_calendar(self) -> List[datetime]:
        """Reject fabricated weekday trading calendars."""
        raise UnsupportedOperationError(
            "Ashare exchange-calendar endpoint is not implemented"
        )
