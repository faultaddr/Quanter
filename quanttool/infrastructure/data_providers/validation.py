"""Strict validation and provenance for market-data boundaries."""

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ...core.errors import DataNotAvailableError, ValidationError


@dataclass(frozen=True)
class DataProvenance:
    """Attribution attached to a validated market-data frame."""

    provider: str
    retrieved_at: datetime
    frequency: str
    adjustment: str
    simulated: bool = False


def validate_market_data(
    frame: pd.DataFrame,
    start_date: datetime,
    end_date: datetime,
    provenance: DataProvenance,
) -> pd.DataFrame:
    """Validate an OHLCV frame and attach immutable source metadata."""
    required = {
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "amount",
    }
    missing = required - set(frame.columns)
    if frame.empty or missing:
        raise ValidationError(
            "Invalid market data: "
            f"empty={frame.empty}, missing={sorted(missing)}"
        )

    result = frame.copy()
    try:
        result["timestamp"] = pd.to_datetime(result["timestamp"])
    except (TypeError, ValueError) as exc:
        raise ValidationError("Market data contains invalid timestamps") from exc

    timestamps = result["timestamp"]
    if not timestamps.is_monotonic_increasing or timestamps.duplicated().any():
        raise ValidationError(
            "Market-data timestamps must be strictly increasing and unique"
        )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    if timestamps.min() < start or timestamps.max() > end:
        raise ValidationError(
            "Market-data timestamp is outside the requested interval"
        )

    numeric = ["open", "high", "low", "close", "volume", "amount"]
    result[numeric] = result[numeric].apply(pd.to_numeric, errors="coerce")
    if result[numeric].isna().any().any() or not np.isfinite(
        result[numeric].to_numpy(dtype=float)
    ).all():
        raise ValidationError("Market data contains non-numeric values")

    if (result[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValidationError("Market data contains non-positive prices")

    valid_ohlc = (
        (result["high"] >= result[["open", "close"]].max(axis=1))
        & (result["low"] <= result[["open", "close"]].min(axis=1))
        & (result["high"] >= result["low"])
    )
    if not valid_ohlc.all():
        raise ValidationError("Market data violates OHLC invariants")

    if (result[["volume", "amount"]] < 0).any().any():
        raise ValidationError("Market data contains negative volume or amount")

    result.attrs["quanttool_provenance"] = asdict(provenance)
    return result


def validate_batch_completeness(
    requested: Sequence[str],
    results: Mapping[str, pd.DataFrame],
    max_missing_ratio: float,
) -> None:
    """Reject batches whose missing-symbol ratio exceeds policy."""
    if not 0.0 <= max_missing_ratio <= 1.0:
        raise ValidationError("max_missing_ratio must be between 0 and 1")

    missing = sorted(set(requested) - set(results))
    ratio = len(missing) / len(requested) if requested else 0.0
    if ratio > max_missing_ratio:
        raise DataNotAvailableError(
            "Market-data batch incomplete: "
            f"missing={missing}, ratio={ratio:.4f}"
        )
