"""Tests for backtest engine."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from quanttool.backtest.engine import BacktestEngine
from quanttool.domain.models import Trade, Order, BacktestResult
from quanttool.strategies.ma_cross import MACrossStrategy


class TestBacktestEngine:
    """Test cases for BacktestEngine."""

    @pytest.fixture
    def sample_data(self):
        """Create sample price data for testing."""
        dates = pd.date_range(start='2023-01-01', end='2023-06-01', freq='D')
        np.random.seed(42)

        # Generate trending price data
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = 100 * np.exp(np.cumsum(returns))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': prices * (1 + np.random.normal(0, 0.005, len(dates))),
            'high': prices * (1 + abs(np.random.normal(0, 0.01, len(dates)))),
            'low': prices * (1 - abs(np.random.normal(0, 0.01, len(dates)))),
            'close': prices,
            'volume': np.random.randint(1000000, 5000000, len(dates)),
            'amount': np.random.randint(10000000, 50000000, len(dates)),
            'timeframe': '1d',
            'symbol': '000001.SZ'
        })

        return {'000001.SZ': df}

    @pytest.fixture
    def engine(self):
        """Create a backtest engine instance."""
        return BacktestEngine()

    @pytest.fixture
    def strategy(self):
        """Create a strategy instance."""
        strategy = MACrossStrategy()
        strategy.initialize({'short_window': 5, 'long_window': 10})
        return strategy

    def test_engine_initialization(self, engine):
        """Test engine initialization."""
        assert engine.initial_cash == 100000.0
        assert engine.commission_rate == 0.0003
        assert engine.slippage_rate == 0.0001

    def test_set_initial_cash(self, engine):
        """Test setting initial cash."""
        engine.set_initial_cash(50000.0)
        assert engine.initial_cash == 50000.0

    def test_set_commission_rate(self, engine):
        """Test setting commission rate."""
        engine.set_commission_rate(0.001)
        assert engine.commission_rate == 0.001

    def test_run_backtest(self, engine, strategy, sample_data):
        """Test running a complete backtest."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 1)

        result = engine.run_backtest(
            strategy=strategy,
            data=sample_data,
            start_date=start_date,
            end_date=end_date
        )

        assert isinstance(result, BacktestResult)
        assert result.start_date == start_date
        assert result.end_date == end_date
        assert result.initial_capital == 100000.0

    def test_backtest_result_metrics(self, engine, strategy, sample_data):
        """Test that backtest result includes all required metrics."""
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 6, 1)

        result = engine.run_backtest(
            strategy=strategy,
            data=sample_data,
            start_date=start_date,
            end_date=end_date
        )

        # Check required fields
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'annual_return')
        assert hasattr(result, 'volatility')
        assert hasattr(result, 'sharpe_ratio')
        assert hasattr(result, 'sortino_ratio')
        assert hasattr(result, 'max_drawdown')
        assert hasattr(result, 'win_rate')
        assert hasattr(result, 'profit_factor')
        assert hasattr(result, 'total_trades')
        assert hasattr(result, 'trades')
        assert hasattr(result, 'equity_curve')

    def test_calculate_metrics_empty_trades(self, engine):
        """Test metrics calculation with no trades."""
        metrics, volatility, sharpe, sortino, max_dd, pf = engine.calculate_metrics([], 100000.0, [])

        assert len(metrics) == 0
        assert volatility == 0.0
        assert sharpe == 0.0
        assert sortino == 0.0
        assert max_dd == 0.0
        assert pf == 0.0

    def test_calculate_metrics_with_trades(self, engine):
        """Test metrics calculation with sample trades."""
        trades = [
            Trade(
                id='1',
                symbol='000001.SZ',
                side='buy',
                quantity=100,
                price=100.0,
                timestamp=datetime.now(),
                fee=5.0,
                pnl=None
            ),
            Trade(
                id='2',
                symbol='000001.SZ',
                side='sell',
                quantity=100,
                price=110.0,
                timestamp=datetime.now(),
                fee=5.5,
                pnl=950.0
            ),
        ]

        equity_curve = [
            {'timestamp': datetime.now(), 'portfolio_value': 100000.0},
            {'timestamp': datetime.now(), 'portfolio_value': 101000.0},
        ]

        metrics, volatility, sharpe, sortino, max_dd, pf = engine.calculate_metrics(
            trades, 100000.0, equity_curve
        )

        assert len(metrics) > 0
        assert pf > 0  # Profit factor should be positive with winning trade

    def test_portfolio_value_calculation(self, engine, sample_data):
        """Test portfolio value calculation."""
        engine.initial_cash = 100000.0
        engine.current_portfolio = type('Portfolio', (), {
            'cash': 50000.0,
            'positions': []
        })()
        engine.positions = {}
        engine.latest_market_prices = {}

        value = engine._calculate_portfolio_value(datetime.now())
        assert value == 50000.0


class TestBacktestEngineEdgeCases:
    """Test edge cases for backtest engine."""

    def test_empty_data(self):
        """Test backtest with empty data."""
        engine = BacktestEngine()
        strategy = MACrossStrategy()
        strategy.initialize({'short_window': 5, 'long_window': 10})

        result = engine.run_backtest(
            strategy=strategy,
            data={},
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 1)
        )

        assert isinstance(result, BacktestResult)
        assert result.total_trades == 0

    def test_single_symbol_data(self):
        """Test backtest with single symbol data."""
        dates = pd.date_range(start='2023-01-01', periods=30, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 30,
            'high': [105.0] * 30,
            'low': [95.0] * 30,
            'close': [100.0] * 30,
            'volume': [1000000] * 30,
            'amount': [100000000] * 30,
            'timeframe': '1d',
            'symbol': '000001.SZ'
        })

        engine = BacktestEngine()
        strategy = MACrossStrategy()
        strategy.initialize({'short_window': 5, 'long_window': 10})

        result = engine.run_backtest(
            strategy=strategy,
            data={'000001.SZ': df},
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 30)
        )

        assert isinstance(result, BacktestResult)
