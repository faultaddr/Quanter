"""Deterministic OHLCV fixtures for algorithm tests."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dates(rows: int) -> pd.DatetimeIndex:
    return pd.date_range("2024-01-02", periods=rows, freq="B")


def _frame_from_close(close: np.ndarray) -> pd.DataFrame:
    rows = len(close)
    dates = _dates(rows)
    idx = np.arange(rows, dtype=float)
    open_ = close * (1 + 0.002 * np.sin(idx / 7.0))
    high = np.maximum(open_, close) * (1 + 0.008 + 0.001 * np.cos(idx / 5.0))
    low = np.minimum(open_, close) * (1 - 0.008 - 0.001 * np.sin(idx / 6.0))
    volume = 1_200_000 + (idx % 21) * 25_000

    df = pd.DataFrame(
        {
            "timestamp": dates,
            "date": dates.strftime("%Y-%m-%d"),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )
    df["amount"] = df["close"] * df["volume"]
    return df


def _add_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close = result["close"]
    high = result["high"]
    low = result["low"]

    for period in [5, 10, 20, 50, 200]:
        result[f"ma_{period}"] = close.rolling(period, min_periods=1).mean()

    mid = close.rolling(20, min_periods=1).mean()
    std = close.rolling(20, min_periods=1).std(ddof=0).fillna(0)
    result["boll_mid"] = mid
    result["boll_upper"] = mid + 2 * std
    result["boll_lower"] = mid - 2 * std

    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr_14"] = tr.rolling(14, min_periods=1).mean()

    delta = close.diff().fillna(0)
    gain = delta.where(delta > 0, 0).rolling(24, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(24, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    result["rsi_24"] = 100 - (100 / (1 + rs))

    highest = high.rolling(14, min_periods=1).max()
    lowest = low.rolling(14, min_periods=1).min()
    result["wr"] = (highest - close) / (highest - lowest + 1e-10) * 100

    typical = (high + low + close) / 3
    typical_ma = typical.rolling(14, min_periods=1).mean()
    mean_dev = typical.rolling(14, min_periods=1).apply(
        lambda x: np.abs(x - x.mean()).mean(),
        raw=True,
    )
    result["cci"] = (typical - typical_ma) / (0.015 * mean_dev.replace(0, np.nan))
    result["cci"] = result["cci"].fillna(0)
    return result


def make_trending_ohlcv(rows: int = 260) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 10.0 + idx * 0.035 + 0.18 * np.sin(idx / 8.0)
    return _frame_from_close(close)


def make_sideways_ohlcv(rows: int = 260) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 12.0 + 0.28 * np.sin(idx / 6.0) + 0.08 * np.cos(idx / 13.0)
    return _frame_from_close(close)


def make_breakout_ohlcv(rows: int = 260) -> pd.DataFrame:
    idx = np.arange(rows, dtype=float)
    close = 9.0 + 0.10 * np.sin(idx / 5.0)
    close[-40:-5] = 9.2 + 0.05 * np.sin(idx[-40:-5] / 3.0)
    close[-5:] = np.linspace(9.45, 10.25, 5)
    return _frame_from_close(close)


def make_indicator_ready_ohlcv(rows: int = 260) -> pd.DataFrame:
    return _add_indicator_columns(make_trending_ohlcv(rows=rows))
