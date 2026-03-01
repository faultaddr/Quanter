"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from quanttool.strategies.ma_cross import MACrossStrategy
from quanttool.strategies.breakout import BreakoutStrategy


class TestMACrossStrategy:
    """Test cases for MA Cross strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a MA Cross strategy instance."""
        strategy = MACrossStrategy()
        strategy.initialize({'short_window': 5, 'long_window': 10})
        return strategy

    @pytest.fixture
    def sample_bars(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 50)))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': close * 0.99,
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 50),
        })
        return df

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.short_window == 5
        assert strategy.long_window == 10
        assert strategy.parameters['short_window'] == 5

    def test_get_name(self, strategy):
        """Test getting strategy name."""
        assert strategy.get_name() == "MA_Cross"

    def test_get_parameters(self, strategy):
        """Test getting parameters."""
        params = strategy.get_parameters()
        assert 'short_window' in params
        assert 'long_window' in params

    def test_get_description(self, strategy):
        """Test getting description."""
        desc = strategy.get_description()
        assert "Moving Average" in desc
        assert "5" in desc
        assert "10" in desc

    def test_calculate_signals(self, strategy, sample_bars):
        """Test signal calculation."""
        signals = strategy.calculate_signals(sample_bars)

        assert isinstance(signals, pd.DataFrame)
        assert 'signal' in signals.columns
        assert 'position' in signals.columns

    def test_get_signal_insufficient_data(self, strategy):
        """Test signal with insufficient data."""
        current_bar = pd.Series({'close': 100.0, 'timestamp': datetime.now()})
        historical_bars = pd.DataFrame({'close': [100.0] * 5})

        signal = strategy.get_signal(current_bar, historical_bars)

        assert signal['direction'] == 'hold'

    def test_get_signal_with_data(self, strategy, sample_bars):
        """Test signal generation with sufficient data."""
        current_bar = sample_bars.iloc[-1]
        historical_bars = sample_bars

        signal = strategy.get_signal(current_bar, historical_bars)

        assert 'direction' in signal
        assert signal['direction'] in ['buy', 'sell', 'hold']

    def test_invalid_parameters(self):
        """Test initialization with invalid parameters."""
        strategy = MACrossStrategy()

        with pytest.raises(ValueError):
            strategy.initialize({'short_window': 20, 'long_window': 10})


class TestBreakoutStrategy:
    """Test cases for Breakout strategy."""

    @pytest.fixture
    def strategy(self):
        """Create a Breakout strategy instance."""
        strategy = BreakoutStrategy()
        strategy.initialize({'lookback_period': 10, 'entry_threshold': 0.02})
        return strategy

    @pytest.fixture
    def sample_bars(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=50, freq='D')
        close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 50)))
        high = close * 1.02
        low = close * 0.98

        df = pd.DataFrame({
            'timestamp': dates,
            'open': close * 0.99,
            'high': high,
            'low': low,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 50),
        })
        return df

    def test_initialization(self, strategy):
        """Test strategy initialization."""
        assert strategy.lookback_period == 10
        assert strategy.entry_threshold == 0.02

    def test_get_name(self, strategy):
        """Test getting strategy name."""
        assert strategy.get_name() == "Breakout"

    def test_calculate_signals(self, strategy, sample_bars):
        """Test signal calculation."""
        signals = strategy.calculate_signals(sample_bars)

        assert isinstance(signals, pd.DataFrame)
        assert 'signal' in signals.columns
        assert 'upper_band' in signals.columns
        assert 'lower_band' in signals.columns

    def test_get_signal_insufficient_data(self, strategy):
        """Test signal with insufficient data."""
        current_bar = pd.Series({'close': 100.0, 'timestamp': datetime.now()})
        historical_bars = pd.DataFrame({'close': [100.0] * 5, 'high': [105.0] * 5, 'low': [95.0] * 5})

        signal = strategy.get_signal(current_bar, historical_bars)

        assert signal['direction'] == 'hold'

    def test_get_signal_breakout_up(self, strategy):
        """Test breakout above upper band."""
        # Create data with clear breakout
        historical_bars = pd.DataFrame({
            'close': [100.0] * 20,
            'high': [105.0] * 20,
            'low': [95.0] * 20
        })
        current_bar = pd.Series({'close': 108.0})  # Above upper band

        signal = strategy.get_signal(current_bar, historical_bars)

        assert signal['direction'] == 'buy'
        assert 'breakout_level' in signal

    def test_get_signal_breakout_down(self, strategy):
        """Test breakout below lower band."""
        # Create data with clear breakdown
        historical_bars = pd.DataFrame({
            'close': [100.0] * 20,
            'high': [105.0] * 20,
            'low': [95.0] * 20
        })
        current_bar = pd.Series({'close': 92.0})  # Below lower band

        signal = strategy.get_signal(current_bar, historical_bars)

        assert signal['direction'] == 'sell'


class TestStrategyEdgeCases:
    """Test edge cases for strategies."""

    def test_ma_cross_empty_data(self):
        """Test MA Cross with empty data."""
        strategy = MACrossStrategy()
        strategy.initialize({'short_window': 5, 'long_window': 10})

        signals = strategy.calculate_signals(pd.DataFrame())
        assert signals.empty

    def test_breakout_empty_data(self):
        """Test Breakout with empty data."""
        strategy = BreakoutStrategy()
        strategy.initialize({'lookback_period': 10, 'entry_threshold': 0.02})

        signals = strategy.calculate_signals(pd.DataFrame())
        assert signals.empty

    def test_ma_cross_single_row(self):
        """Test MA Cross with single row."""
        strategy = MACrossStrategy()
        strategy.initialize({'short_window': 5, 'long_window': 10})

        df = pd.DataFrame({'close': [100.0]})
        signals = strategy.calculate_signals(df)

        assert len(signals) == 1
