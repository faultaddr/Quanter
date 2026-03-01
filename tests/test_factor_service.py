"""Tests for factor service."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock

from quanttool.application.factor_service import FactorService
from quanttool.domain.models import FactorEvaluationResult


class TestFactorService:
    """Test cases for FactorService."""

    @pytest.fixture
    def factor_service(self):
        """Create a factor service instance."""
        return FactorService()

    @pytest.fixture
    def sample_bars(self):
        """Create sample price data."""
        np.random.seed(42)
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        close = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))

        df = pd.DataFrame({
            'timestamp': dates,
            'open': close * 0.99,
            'high': close * 1.02,
            'low': close * 0.98,
            'close': close,
            'volume': np.random.randint(1000000, 5000000, 100),
            'amount': np.random.randint(10000000, 50000000, 100),
        })
        return df

    def test_initialization(self, factor_service):
        """Test factor service initialization."""
        assert factor_service is not None

    def test_calculate_ic(self, factor_service):
        """Test IC calculation."""
        factor_values = pd.Series([1, 2, 3, 4, 5])
        returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

        ic = factor_service._calculate_ic(factor_values, returns)

        assert isinstance(ic, float)
        assert abs(ic) <= 1.0

    def test_calculate_rank_ic(self, factor_service):
        """Test rank IC calculation."""
        factor_values = pd.Series([1, 2, 3, 4, 5])
        returns = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])

        rank_ic = factor_service._calculate_rank_ic(factor_values, returns)

        assert isinstance(rank_ic, float)
        assert abs(rank_ic) <= 1.0

    def test_calculate_ic_empty(self, factor_service):
        """Test IC calculation with empty data."""
        factor_values = pd.Series([])
        returns = pd.Series([])

        ic = factor_service._calculate_ic(factor_values, returns)

        assert ic == 0.0

    def test_calculate_ic_with_nans(self, factor_service):
        """Test IC calculation with NaN values."""
        factor_values = pd.Series([1, 2, np.nan, 4, 5])
        returns = pd.Series([0.1, 0.2, 0.3, np.nan, 0.5])

        ic = factor_service._calculate_ic(factor_values, returns)

        assert isinstance(ic, float)

    def test_mine_factor_mock_data(self, factor_service, sample_bars, monkeypatch):
        """Test factor mining with mock data."""
        # Mock the data provider
        mock_provider = Mock()
        mock_provider.get_bars.return_value = {'000001.SZ': sample_bars}
        mock_provider.initialize = Mock()

        # Mock registry.get
        def mock_get(component_type, name):
            if name == 'momentum':
                from quanttool.factors.technical.momentum import MomentumFactor
                return MomentumFactor
            return mock_provider

        monkeypatch.setattr('quanttool.application.factor_service.registry.get', mock_get)

        results = factor_service.mine_factor(
            factor_name='momentum',
            factor_params={'period': 10},
            symbols=['000001.SZ'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 6, 1),
            data_provider='mock'
        )

        assert isinstance(results, dict)
        assert '000001.SZ' in results
        result = results['000001.SZ']
        assert isinstance(result, FactorEvaluationResult)
        assert hasattr(result, 'ic')
        assert hasattr(result, 'rank_ic')
        assert hasattr(result, 'win_rate')


class TestFactorServiceMetrics:
    """Test factor service metric calculations."""

    @pytest.fixture
    def service(self):
        return FactorService()

    def test_ic_perfect_correlation(self, service):
        """Test IC with perfect correlation."""
        factor = pd.Series([1, 2, 3, 4, 5])
        returns = pd.Series([1, 2, 3, 4, 5])

        ic = service._calculate_ic(factor, returns)
        assert abs(ic - 1.0) < 0.01

    def test_ic_perfect_negative_correlation(self, service):
        """Test IC with perfect negative correlation."""
        factor = pd.Series([1, 2, 3, 4, 5])
        returns = pd.Series([5, 4, 3, 2, 1])

        ic = service._calculate_ic(factor, returns)
        assert abs(ic - (-1.0)) < 0.01

    def test_ic_no_correlation(self, service):
        """Test IC with no correlation."""
        np.random.seed(42)
        factor = pd.Series(np.random.randn(100))
        returns = pd.Series(np.random.randn(100))

        ic = service._calculate_ic(factor, returns)
        assert abs(ic) < 0.3  # Should be close to 0
