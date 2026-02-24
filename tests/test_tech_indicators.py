"""Tests for technical indicators."""

import pytest
import numpy as np
import pandas as pd

from quanttool.factors.tech_indicators import (
    MA, EMA, RSI, MACD, KDJ, BOLL, BIAS, CCI,
    ATR, DMI, TRIX, VR, CR, WR, BBI, PSY,
    LLV, HHV
)


class TestMovingAverages:
    """Test moving average indicators."""

    def test_ma_basic(self):
        """Test basic MA calculation."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = MA(data, 5)

        assert len(result) == len(data)
        assert np.isnan(result[0])  # First values should be NaN
        assert result[4] == 3.0  # Average of [1,2,3,4,5]
        assert result[9] == 8.0  # Average of [6,7,8,9,10]

    def test_ema_basic(self):
        """Test basic EMA calculation."""
        data = np.array([100.0] * 10)
        result = EMA(data, 5)

        assert len(result) == len(data)
        # EMA of constant series should be constant
        assert result[-1] == 100.0


class TestRSI:
    """Test RSI indicator."""

    def test_rsi_constant_price(self):
        """Test RSI with constant price."""
        data = np.array([100.0] * 20)
        result = RSI(data, 14)

        assert len(result) == len(data)
        # RSI of constant price should be around 50
        assert not np.isnan(result[-1])

    def test_rsi_uptrend(self):
        """Test RSI in uptrend."""
        data = np.array([100 + i for i in range(30)])
        result = RSI(data, 14)

        # RSI should be high in uptrend
        assert result[-1] > 50

    def test_rsi_downtrend(self):
        """Test RSI in downtrend."""
        data = np.array([100 - i for i in range(30)])
        result = RSI(data, 14)

        # RSI should be low in downtrend
        assert result[-1] < 50

    def test_rsi_range(self):
        """Test RSI values are in valid range."""
        np.random.seed(42)
        data = 100 * (1 + np.random.normal(0, 0.02, 100)).cumsum()
        result = RSI(data, 14)

        valid_values = result[~np.isnan(result)]
        assert all(0 <= v <= 100 for v in valid_values)


class TestMACD:
    """Test MACD indicator."""

    def test_macd_basic(self):
        """Test MACD calculation."""
        data = np.array([100 + i for i in range(50)])
        dif, dea, macd = MACD(data)

        assert len(dif) == len(data)
        assert len(dea) == len(data)
        assert len(macd) == len(data)

    def test_macd_trending(self):
        """Test MACD in trending market."""
        # Strong uptrend
        data = np.array([100 + i * 2 for i in range(50)])
        dif, dea, macd = MACD(data)

        # DIF should be above DEA in strong uptrend
        assert dif[-1] > dea[-1]


class TestBollingerBands:
    """Test Bollinger Bands indicator."""

    def test_boll_basic(self):
        """Test Bollinger Bands calculation."""
        np.random.seed(42)
        data = 100 * (1 + np.random.normal(0, 0.02, 50)).cumsum()
        upper, mid, lower = BOLL(data)

        assert len(upper) == len(data)
        assert len(mid) == len(data)
        assert len(lower) == len(data)

    def test_boll_relationship(self):
        """Test Bollinger Bands relationships."""
        np.random.seed(42)
        data = 100 * (1 + np.random.normal(0, 0.02, 50)).cumsum()
        upper, mid, lower = BOLL(data)

        # Upper > Middle > Lower
        valid_idx = ~np.isnan(upper)
        assert all(upper[valid_idx] >= mid[valid_idx])
        assert all(mid[valid_idx] >= lower[valid_idx])


class TestKDJ:
    """Test KDJ indicator."""

    def test_kdj_basic(self):
        """Test KDJ calculation."""
        np.random.seed(42)
        close = 100 * (1 + np.random.normal(0, 0.02, 50)).cumsum()
        high = close * 1.02
        low = close * 0.98

        k, d, j = KDJ(close, high, low)

        assert len(k) == len(close)
        assert len(d) == len(close)
        assert len(j) == len(close)


class TestATR:
    """Test ATR indicator."""

    def test_atr_basic(self):
        """Test ATR calculation."""
        np.random.seed(42)
        close = 100 * (1 + np.random.normal(0, 0.02, 50)).cumsum()
        high = close * 1.02
        low = close * 0.98

        result = ATR(close, high, low, 14)

        assert len(result) == len(close)
        assert all(result[~np.isnan(result)] >= 0)  # ATR should be positive


class TestLLVHHV:
    """Test LLV and HHV functions."""

    def test_llv_basic(self):
        """Test Lowest Low Value."""
        data = np.array([5, 4, 3, 2, 1, 2, 3, 4, 5])
        result = LLV(data, 3)

        assert len(result) == len(data)
        assert result[2] == 3  # Min of [5,4,3]
        assert result[4] == 1  # Min of [3,2,1]

    def test_hhv_basic(self):
        """Test Highest High Value."""
        data = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1])
        result = HHV(data, 3)

        assert len(result) == len(data)
        assert result[2] == 3  # Max of [1,2,3]
        assert result[4] == 5  # Max of [3,4,5]


class TestVolatilityIndicators:
    """Test volatility indicators."""

    def test_cci_basic(self):
        """Test CCI calculation."""
        np.random.seed(42)
        close = 100 * (1 + np.random.normal(0, 0.02, 50)).cumsum()
        high = close * 1.02
        low = close * 0.98

        result = CCI(close, high, low)

        assert len(result) == len(close)

    def test_bias_basic(self):
        """Test BIAS calculation."""
        data = np.array([100 + i for i in range(50)])
        bias6, bias12, bias24 = BIAS(data)

        assert len(bias6) == len(data)
        assert len(bias12) == len(data)
        assert len(bias24) == len(data)


class TestVolumeIndicators:
    """Test volume-based indicators."""

    def test_vr_basic(self):
        """Test VR calculation."""
        np.random.seed(42)
        close = 100 * (1 + np.random.normal(0, 0.02, 50)).cumsum()
        volume = np.random.randint(1000000, 5000000, 50)

        result = VR(close, volume)

        assert len(result) == len(close)

    def test_cr_basic(self):
        """Test CR calculation."""
        np.random.seed(42)
        close = 100 * (1 + np.random.normal(0, 0.02, 50)).cumsum()
        high = close * 1.02
        low = close * 0.98

        result = CR(close, high, low)

        assert len(result) == len(close)


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_array(self):
        """Test with empty array."""
        data = np.array([])
        result = MA(data, 5)
        assert len(result) == 0

    def test_single_value(self):
        """Test with single value."""
        data = np.array([100.0])
        result = MA(data, 5)
        assert len(result) == 1
        assert np.isnan(result[0])

    def test_all_nan(self):
        """Test with all NaN values."""
        data = np.array([np.nan] * 10)
        result = MA(data, 5)
        assert all(np.isnan(result))

    def test_period_larger_than_data(self):
        """Test when period is larger than data length."""
        data = np.array([1, 2, 3])
        result = MA(data, 10)
        assert all(np.isnan(result))
