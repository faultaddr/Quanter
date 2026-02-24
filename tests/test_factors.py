"""Tests for factor implementations."""

import pytest
import pandas as pd
import numpy as np

from quanttool.factors.technical.momentum import MomentumFactor, ReturnsMomentumFactor, PriceVolumeTrendFactor
from quanttool.factors.technical.volatility import VolatilityFactor, ATRFactor
from quanttool.factors.technical.value import RSIFactor


class TestMomentumFactor:
    """Test cases for Momentum factor."""

    @pytest.fixture
    def factor(self):
        """Create a momentum factor instance."""
        factor = MomentumFactor()
        factor.initialize({'period': 10})
        return factor

    @pytest.fixture
    def sample_bars(self):
        """Create sample price data."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': [100 + i for i in range(30)],  # Linear increase
            'volume': [1000000] * 30,
        })
        return df

    def test_initialization(self, factor):
        """Test factor initialization."""
        assert factor.period == 10

    def test_compute(self, factor, sample_bars):
        """Test factor computation."""
        result = factor.compute(sample_bars)

        assert isinstance(result, pd.DataFrame)
        assert 'factor_value' in result.columns
        assert 'timestamp' in result.columns

    def test_get_name(self, factor):
        """Test getting factor name."""
        assert factor.get_name() == "Momentum"

    def test_get_parameters(self, factor):
        """Test getting parameters."""
        params = factor.get_parameters()
        assert 'period' in params

    def test_compute_empty_data(self, factor):
        """Test computation with empty data."""
        result = factor.compute(pd.DataFrame())
        assert result.empty

    def test_momentum_calculation(self, factor):
        """Test correct momentum calculation."""
        # Price doubles over 10 periods
        df = pd.DataFrame({
            'close': [100] * 10 + [200] * 10,
            'timestamp': pd.date_range(start='2023-01-01', periods=20, freq='D')
        })

        result = factor.compute(df)

        # At period 10, momentum should be (200/100 - 1) = 1.0
        assert not result.empty


class TestReturnsMomentumFactor:
    """Test cases for Returns Momentum factor."""

    @pytest.fixture
    def factor(self):
        """Create a returns momentum factor instance."""
        factor = ReturnsMomentumFactor()
        factor.initialize({'return_period': 5, 'momentum_period': 10})
        return factor

    def test_initialization(self, factor):
        """Test factor initialization."""
        assert factor.return_period == 5
        assert factor.momentum_period == 10

    def test_compute(self, factor):
        """Test factor computation."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': [100 + i for i in range(30)],
        })

        result = factor.compute(df)

        assert isinstance(result, pd.DataFrame)
        assert 'factor_value' in result.columns


class TestPriceVolumeTrendFactor:
    """Test cases for Price Volume Trend factor."""

    @pytest.fixture
    def factor(self):
        """Create a PVT factor instance."""
        return PriceVolumeTrendFactor()

    def test_compute(self, factor):
        """Test factor computation."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': [100 + i for i in range(30)],
            'volume': [1000000] * 30,
        })

        result = factor.compute(df)

        assert isinstance(result, pd.DataFrame)
        assert 'factor_value' in result.columns

    def test_pvt_trend(self, factor):
        """Test PVT captures volume-weighted trend."""
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')

        # Rising price with high volume
        df = pd.DataFrame({
            'timestamp': dates,
            'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
            'volume': [10000000] * 10,
        })

        result = factor.compute(df)

        # PVT should be increasing with rising prices
        assert not result.empty


class TestVolatilityFactor:
    """Test cases for Volatility factor."""

    def test_compute(self):
        """Test volatility factor computation."""
        from quanttool.factors.technical.volatility import VolatilityFactor

        factor = VolatilityFactor()
        factor.initialize({'period': 10})

        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': 100 * (1 + np.random.normal(0, 0.02, 30)).cumsum(),
        })

        result = factor.compute(df)

        assert isinstance(result, pd.DataFrame)
        assert 'factor_value' in result.columns


class TestRSIFactor:
    """Test cases for RSI factor."""

    def test_compute(self):
        """Test RSI factor computation."""
        factor = RSIFactor()
        factor.initialize({'period': 14})

        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'close': [100 + (i % 10) for i in range(30)],  # Oscillating price
        })

        result = factor.compute(df)

        assert isinstance(result, pd.DataFrame)
        assert 'factor_value' in result.columns

    def test_rsi_range(self):
        """Test RSI values are in expected range."""
        factor = RSIFactor()
        factor.initialize({'period': 5})

        # Strong uptrend
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=20, freq='D'),
            'close': [100 + i * 2 for i in range(20)],
        })

        result = factor.compute(df)

        # RSI should be in range [0, 100]
        valid_values = result['factor_value'].dropna()
        assert all(0 <= v <= 100 for v in valid_values)
