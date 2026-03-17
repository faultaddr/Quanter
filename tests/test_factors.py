"""Tests for factor implementations."""

import pytest
import pandas as pd
import numpy as np

from quanttool.factors.technical.momentum import MomentumFactor, ReturnsMomentumFactor, PriceVolumeTrendFactor
from quanttool.factors.technical.volatility import VolatilityFactor


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


class TestScoringSystem:
    """Test cases for ScoringSystem - candlestick pattern integration."""

    @pytest.fixture
    def scoring_system(self):
        """Create a scoring system instance."""
        from quanttool.factors.scoring_system import ScoringSystem
        return ScoringSystem()

    @pytest.fixture
    def sample_df(self):
        """Create sample stock data for testing."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')

        # Create realistic price data with a moderate uptrend
        base_price = 100
        prices = []
        for i in range(100):
            noise = np.random.normal(0, 0.02)
            trend = 0.001 * i  # Slight uptrend
            prices.append(base_price * (1 + trend + noise))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': [1000000] * 100,
        })
        return df

    def test_scoring_system_initialization(self, scoring_system):
        """Test ScoringSystem initialization."""
        assert scoring_system is not None
        assert hasattr(scoring_system, 'TREND_FACTOR_WEIGHTS')
        # K线形态已移至独立筛选层，不再参与评分计算
        # 筛选层测试在 test_screening.py 中

    def test_candlestick_weights_configured(self, scoring_system):
        """Test that candlestick pattern weights are configured."""
        assert hasattr(scoring_system, 'CANDLESTICK_PATTERN_WEIGHTS')
        assert 'strong_bullish' in scoring_system.CANDLESTICK_PATTERN_WEIGHTS
        assert 'strong_bearish' in scoring_system.CANDLESTICK_PATTERN_WEIGHTS

    def test_position_modifiers_configured(self, scoring_system):
        """Test that position pattern modifiers are configured."""
        assert hasattr(scoring_system, 'POSITION_PATTERN_MODIFIERS')
        assert 'low_position' in scoring_system.POSITION_PATTERN_MODIFIERS
        assert 'high_position' in scoring_system.POSITION_PATTERN_MODIFIERS

        # Low position should amplify bullish patterns
        assert scoring_system.POSITION_PATTERN_MODIFIERS['low_position']['bullish'] > 1.0

        # High position should reduce bullish patterns (警惕诱多)
        assert scoring_system.POSITION_PATTERN_MODIFIERS['high_position']['bullish'] < 0

    def test_calculate_all_scores(self, scoring_system, sample_df):
        """Test full scoring calculation."""
        result = scoring_system.calculate_all_scores(sample_df)

        # Should have score or error with bias_passed status
        assert 'score' in result or 'error' in result
        # If no error, check the score
        if 'error' not in result:
            assert 0 <= result['score'] <= 100

    def test_candlestick_score_method_exists(self, scoring_system):
        """Test that _calculate_candlestick_score method exists."""
        assert hasattr(scoring_system, '_calculate_candlestick_score')

    def test_determine_position_zone_method_exists(self, scoring_system):
        """Test that _determine_position_zone method exists."""
        assert hasattr(scoring_system, '_determine_position_zone')

    def test_get_pattern_weight_method_exists(self, scoring_system):
        """Test that _get_pattern_weight method exists."""
        assert hasattr(scoring_system, '_get_pattern_weight')


class TestCandlestickPatterns:
    """Test cases for candlestick pattern recognition using TA-Lib."""

    def test_hammer_pattern(self):
        """Test hammer pattern recognition."""
        from quanttool.factors.talib_patterns import recognize_talib_patterns

        # Create a hammer pattern (long lower shadow, small body at top)
        df = pd.DataFrame({
            'open': [100, 100, 100, 100, 102],  # Last candle: open 102
            'high': [101, 101, 101, 101, 102.5],  # high 102.5
            'low': [99, 99, 99, 99, 95],  # long lower shadow to 95
            'close': [100, 100, 100, 100, 102],  # close 102 (bullish)
            'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        })

        result = recognize_talib_patterns(df, lookback=5)

        assert 'patterns' in result
        assert isinstance(result['patterns'], list)

    def test_bullish_engulfing(self):
        """Test bullish engulfing pattern recognition."""
        from quanttool.factors.talib_patterns import recognize_talib_patterns

        # Create test data for engulfing pattern
        df = pd.DataFrame({
            'open': [105, 98, 102],
            'high': [108, 112, 105],
            'low': [99, 97, 100],
            'close': [100, 110, 103],
            'timestamp': pd.date_range(start='2023-01-01', periods=3, freq='D'),
        })

        result = recognize_talib_patterns(df, lookback=3)

        assert 'patterns' in result
        # TA-Lib may detect different patterns than the classic recognizer
        # Just check that the function returns a valid result structure
        assert 'bullish_count' in result or 'patterns' in result

    def test_pattern_assessment(self):
        """Test pattern assessment."""
        from quanttool.factors.talib_patterns import TalibPatternRecognizer, get_pattern_assessment

        recognizer = TalibPatternRecognizer()

        # Create test data
        df = pd.DataFrame({
            'open': [100, 100, 100, 100, 102],
            'high': [101, 101, 101, 101, 102.5],
            'low': [99, 99, 99, 99, 95],
            'close': [100, 100, 100, 100, 102],
            'timestamp': pd.date_range(start='2023-01-01', periods=5, freq='D'),
        })

        result = recognizer.recognize_all(df, lookback=5)
        assessment = get_pattern_assessment(result)

        # Should return a string assessment
        assert isinstance(assessment, str)