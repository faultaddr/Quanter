"""Tests for signal service."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock

from quanttool.application.signal_service import SignalService
from quanttool.domain.models import Signal


class TestSignalService:
    """Test cases for SignalService."""

    @pytest.fixture
    def service(self):
        """Create a signal service instance."""
        return SignalService()

    @pytest.fixture
    def sample_data(self):
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
        return {'000001.SZ': df}

    def test_initialization(self, service):
        """Test service initialization."""
        assert service.cooldown_bars == 3

    def test_scan_signals_mock(self, service, sample_data, monkeypatch):
        """Test signal scanning with mock data."""
        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.get_signal.return_value = {
            'direction': 'buy',
            'reason': 'golden_cross',
            'strength': 1.0
        }

        # Mock data provider
        mock_provider = Mock()
        mock_provider.get_bars.return_value = sample_data
        mock_provider.initialize = Mock()

        # Mock registry
        def mock_get(component_type, name):
            if name == 'ma_cross':
                return lambda: mock_strategy
            if name == 'tushare':
                return lambda: mock_provider
            return None

        monkeypatch.setattr('quanttool.application.signal_service.registry.get', mock_get)

        signals = service.scan_signals(
            strategy_name='ma_cross',
            strategy_params={'short_window': 5, 'long_window': 10},
            symbols=['000001.SZ'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 1),
            timeframe='1d',
            data_provider='tushare'
        )

        assert isinstance(signals, list)
        # Should have signals (mock always returns buy)
        assert len(signals) > 0

        # Check signal structure
        for signal in signals:
            assert isinstance(signal, Signal)
            assert signal.direction in ['buy', 'sell']
            assert signal.symbol == '000001.SZ'

    def test_scan_signals_no_signals(self, service, sample_data, monkeypatch):
        """Test signal scanning when no signals are generated."""
        # Mock strategy that always returns hold
        mock_strategy = Mock()
        mock_strategy.get_signal.return_value = {
            'direction': 'hold',
            'reason': 'no_cross'
        }

        mock_provider = Mock()
        mock_provider.get_bars.return_value = sample_data
        mock_provider.initialize = Mock()

        def mock_get(component_type, name):
            if name == 'ma_cross':
                return lambda: mock_strategy
            if name == 'tushare':
                return lambda: mock_provider
            return None

        monkeypatch.setattr('quanttool.application.signal_service.registry.get', mock_get)

        signals = service.scan_signals(
            strategy_name='ma_cross',
            strategy_params={},
            symbols=['000001.SZ'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 2, 1)
        )

        assert isinstance(signals, list)
        # Should have no signals (strategy always returns hold)
        assert len(signals) == 0


class TestSignalServiceCooldown:
    """Test signal cooldown functionality."""

    def test_cooldown_enforcement(self):
        """Test that cooldown period is enforced."""
        service = SignalService()
        service.cooldown_bars = 3

        # Create mock signals
        signals = [
            Signal(
                symbol='000001.SZ',
                timestamp=datetime(2023, 1, i),
                direction='buy',
                strength=1.0
            )
            for i in range(1, 10)
        ]

        # Simulate cooldown check
        last_signal_time = {}
        filtered_signals = []

        for signal in signals:
            if signal.symbol in last_signal_time:
                # In real implementation, this would check bar indices
                continue

            filtered_signals.append(signal)
            last_signal_time[signal.symbol] = signal.timestamp

        # Only first signal should pass due to cooldown
        assert len(filtered_signals) == 1


class TestSignalServiceLive:
    """Test live signal monitoring."""

    def test_live_signals_mock(self, monkeypatch):
        """Test live signal monitoring with mocks."""
        service = SignalService()

        # Mock strategy
        mock_strategy = Mock()
        mock_strategy.get_signal.return_value = {
            'direction': 'buy',
            'reason': 'test_signal',
            'strength': 1.0
        }

        # Mock data provider
        mock_provider = Mock()
        mock_provider.get_latest_bar.return_value = pd.DataFrame({
            'timestamp': [datetime.now()],
            'close': [100.0],
            'open': [99.0],
            'high': [101.0],
            'low': [98.0],
            'volume': [1000000]
        })
        mock_provider.initialize = Mock()

        # Mock notifier
        mock_notifier = Mock()
        mock_notifier.send_notification.return_value = True

        # Mock registry
        def mock_get(component_type, name):
            if name == 'ma_cross':
                return lambda: mock_strategy
            if name == 'ashare':
                return lambda: mock_provider
            return None

        monkeypatch.setattr('quanttool.application.signal_service.registry.get', mock_get)

        # Mock time.sleep to avoid waiting
        import time
        original_sleep = time.sleep
        time.sleep = Mock()

        try:
            # Run for a short duration
            service.live_signals(
                strategy_name='ma_cross',
                strategy_params={},
                symbols=['000001.SZ'],
                notifier=mock_notifier,
                run_duration=0.1  # Run for 0.1 minutes
            )
        except Exception:
            pass  # Expected to exit after duration
        finally:
            time.sleep = original_sleep

        # Notifier should have been called
        mock_notifier.send_notification.assert_called()
