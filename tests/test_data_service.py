"""Tests for data service."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import pandas as pd

from quanttool.application.data_service import DataService
from quanttool.domain.interfaces.data_provider import IDataProvider
from quanttool.domain.interfaces.store import IStore


class TestDataService:
    """Test cases for DataService."""

    @pytest.fixture
    def data_service(self):
        """Create a data service instance."""
        return DataService()

    @pytest.fixture
    def mock_provider(self):
        """Create a mock data provider."""
        provider = Mock(spec=IDataProvider)

        # Mock get_bars
        dates = pd.date_range(start='2023-01-01', periods=10, freq='D')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [100.0] * 10,
            'volume': [1000000] * 10,
            'amount': [100000000] * 10,
            'timeframe': '1d',
            'symbol': '000001.SZ'
        })
        provider.get_bars.return_value = {'000001.SZ': df}

        # Mock other methods
        provider.get_calendar.return_value = [datetime(2023, 1, i) for i in range(1, 10)]
        provider.search_symbols.return_value = [{'symbol': '000001.SZ', 'name': 'Test'}]
        provider.get_latest_bar.return_value = df.iloc[[-1]]

        return provider

    @pytest.fixture
    def mock_store(self):
        """Create a mock store."""
        store = Mock(spec=IStore)
        store.save_data.return_value = True
        return store

    def test_initialization(self, data_service):
        """Test data service initialization."""
        assert data_service.data_providers == {}
        assert data_service.store is None

    def test_register_data_provider(self, data_service, mock_provider):
        """Test registering a data provider."""
        data_service.register_data_provider('test', mock_provider)
        assert 'test' in data_service.data_providers
        assert data_service.data_providers['test'] == mock_provider

    def test_set_store(self, data_service, mock_store):
        """Test setting the store."""
        data_service.set_store(mock_store)
        assert data_service.store == mock_store

    def test_get_data_provider_registered(self, data_service, mock_provider):
        """Test getting a registered data provider."""
        data_service.register_data_provider('test', mock_provider)
        provider = data_service.get_data_provider('test')
        assert provider == mock_provider

    def test_pull_data_without_store(self, data_service, mock_provider):
        """Test pulling data without saving to store."""
        data_service.register_data_provider('test', mock_provider)

        result = data_service.pull_data(
            provider_name='test',
            symbols=['000001.SZ'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            timeframe='1d',
            save_to_store=False
        )

        assert 'data' in result
        assert 'provider' in result
        assert result['provider'] == 'test'
        assert '000001.SZ' in result['data']

    def test_pull_data_with_store(self, data_service, mock_provider, mock_store):
        """Test pulling data and saving to store."""
        data_service.register_data_provider('test', mock_provider)
        data_service.set_store(mock_store)

        result = data_service.pull_data(
            provider_name='test',
            symbols=['000001.SZ'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            timeframe='1d',
            save_to_store=True
        )

        assert 'data' in result
        mock_store.save_data.assert_called_once()

    def test_get_calendar(self, data_service, mock_provider):
        """Test getting trading calendar."""
        data_service.register_data_provider('test', mock_provider)

        calendar = data_service.get_calendar('test')

        assert len(calendar) > 0
        mock_provider.get_calendar.assert_called_once()

    def test_search_symbols(self, data_service, mock_provider):
        """Test searching symbols."""
        data_service.register_data_provider('test', mock_provider)

        results = data_service.search_symbols('test', 'ping')

        assert len(results) > 0
        mock_provider.search_symbols.assert_called_once_with('ping')

    def test_get_latest_bar(self, data_service, mock_provider):
        """Test getting latest bar."""
        data_service.register_data_provider('test', mock_provider)

        bar = data_service.get_latest_bar('test', '000001.SZ')

        assert bar is not None
        mock_provider.get_latest_bar.assert_called_once_with('000001.SZ', '1d')


class TestDataServiceIntegration:
    """Integration tests for DataService."""

    @pytest.mark.integration
    def test_pull_data_csv_provider(self):
        """Test pulling data from CSV provider."""
        from quanttool.infrastructure.data_providers.csv_provider import CSVProvider

        service = DataService()
        provider = CSVProvider('./mock_data')
        provider.initialize()

        service.register_data_provider('csv', provider)

        result = service.pull_data(
            provider_name='csv',
            symbols=['000001.SZ'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 31),
            timeframe='1d',
            save_to_store=False
        )

        assert 'data' in result
