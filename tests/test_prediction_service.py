"""Tests for prediction service."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from unittest.mock import Mock, patch

from quanttool.application.prediction_service import PredictionService


# 定义在模块级别的 fixture，供所有测试类使用
@pytest.fixture
def service():
    """Create a prediction service instance."""
    return PredictionService()


@pytest.fixture
def sample_bars():
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


class TestPredictionService:
    """Test cases for PredictionService."""

    def test_initialization(self, service):
        """Test service initialization."""
        assert service.default_horizon == 6

    def test_prepare_features(self, service, sample_bars):
        """Test feature preparation."""
        features_df = service.prepare_features(sample_bars, horizon=6)

        assert isinstance(features_df, pd.DataFrame)
        assert 'target' in features_df.columns

        # Check expected feature columns exist
        expected_features = [
            'returns_lag1', 'returns_lag2', 'ma_5', 'ma_10', 'ma_20',
            'rsi', 'bb_position', 'volume_ratio', 'volatility'
        ]
        for feature in expected_features:
            assert feature in features_df.columns

    def test_prepare_features_empty_data(self, service):
        """Test feature preparation with empty data."""
        empty_df = pd.DataFrame()
        # 空数据应该返回空 DataFrame 或抛出异常
        try:
            result = service.prepare_features(empty_df)
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 0
        except KeyError:
            # 空数据没有必要的列，这是预期行为
            pass

    def test_train_model_mock(self, service, sample_bars, monkeypatch):
        """Test model training with mock data."""
        # Mock data provider
        mock_provider = Mock()
        mock_provider.get_bars.return_value = {'000001.SZ': sample_bars}
        mock_provider.initialize = Mock()

        # Mock registry
        def mock_get(component_type, name):
            if name == 'tushare':
                return lambda: mock_provider
            return None

        monkeypatch.setattr('quanttool.application.prediction_service.registry.get', mock_get)

        # Mock sklearn to avoid actual training
        with patch('sklearn.linear_model.LogisticRegression') as mock_lr:
            mock_model = Mock()
            mock_model.fit = Mock()
            mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
            mock_lr.return_value = mock_model

            result = service.train_model(
                symbol='000001.SZ',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 6, 1),
                horizon=6,
                model_type='logistic_regression',
                data_provider='tushare',
                timeframe='1d'
            )

            assert 'model_id' in result
            assert 'symbol' in result
            assert 'metrics' in result
            assert result['symbol'] == '000001.SZ'

    def test_predict_mock(self, service, sample_bars, monkeypatch):
        """Test prediction with mock data."""
        # Mock data provider
        mock_provider = Mock()
        mock_provider.get_bars.return_value = {'000001.SZ': sample_bars}
        mock_provider.initialize = Mock()

        def mock_get(component_type, name):
            if name == 'tushare':
                return lambda: mock_provider
            return None

        monkeypatch.setattr('quanttool.application.prediction_service.registry.get', mock_get)

        # Mock sklearn
        with patch('sklearn.linear_model.LogisticRegression') as mock_lr:
            mock_model = Mock()
            mock_model.fit = Mock()
            mock_model.predict_proba = Mock(return_value=np.array([[0.3, 0.7]]))
            mock_model.predict = Mock(return_value=np.array([1]))
            mock_model.coef_ = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]])
            mock_lr.return_value = mock_model

            result = service.predict(
                model_id='test-model',
                symbol='000001.SZ',
                data=sample_bars,
                data_provider='tushare',
                timeframe='1d'
            )

            assert 'prediction' in result
            assert 'probability_positive' in result
            assert 'probability_negative' in result
            assert 'feature_importance' in result
            assert result['symbol'] == '000001.SZ'


class TestPredictionServiceEdgeCases:
    """Test edge cases for prediction service."""

    def test_train_model_insufficient_data(self):
        """Test training with insufficient data."""
        service = PredictionService()

        # Create minimal data
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'close': [100.0] * 10,
            'volume': [1000000] * 10,
        })

        mock_provider = Mock()
        mock_provider.get_bars.return_value = {'000001.SZ': df}
        mock_provider.initialize = Mock()

        with pytest.raises(ValueError):
            with patch('quanttool.application.prediction_service.registry.get',
                      return_value=lambda: mock_provider):
                service.train_model(
                    symbol='000001.SZ',
                    start_date=datetime(2023, 1, 1),
                    end_date=datetime(2023, 1, 10)
                )

    def test_unsupported_model_type(self, service, sample_bars, monkeypatch):
        """Test with unsupported model type."""
        mock_provider = Mock()
        mock_provider.get_bars.return_value = {'000001.SZ': sample_bars}
        mock_provider.initialize = Mock()

        def mock_get(component_type, name):
            if name == 'tushare':
                return lambda: mock_provider
            return None

        monkeypatch.setattr('quanttool.application.prediction_service.registry.get', mock_get)

        with pytest.raises(ValueError) as exc_info:
            service.train_model(
                symbol='000001.SZ',
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 6, 1),
                model_type='unsupported_model'
            )

        assert 'Unsupported model type' in str(exc_info.value)
