"""Tests for data providers."""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import tempfile
import shutil
from pathlib import Path

from quanttool.infrastructure.data_providers.csv_provider import CSVProvider
from quanttool.infrastructure.data_providers.tushare_provider import TuShareProvider
from quanttool.infrastructure.data_providers.ashare_provider import AShareProvider


class TestCSVProvider:
    """Test cases for CSVProvider."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory."""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def provider(self, temp_dir):
        """Create a CSVProvider instance."""
        return CSVProvider(temp_dir)

    def test_initialization(self, temp_dir):
        """Test provider initialization."""
        provider = CSVProvider(temp_dir)
        provider.initialize()
        assert provider._initialized is True

    def test_get_supported_symbols(self, provider):
        """Test getting supported symbols."""
        provider.initialize()

        # Create a sample CSV file
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'open': [100.0] * 10,
            'high': [105.0] * 10,
            'low': [95.0] * 10,
            'close': [100.0] * 10,
            'volume': [1000000] * 10,
            'amount': [100000000] * 10,
        })
        df.to_csv(Path(provider.data_dir) / '000001.SZ.csv', index=False)

        symbols = provider.get_supported_symbols()

        assert '000001.SZ' in symbols

    def test_get_bars(self, provider):
        """Test getting bars."""
        provider.initialize()

        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=30, freq='D'),
            'open': [100.0 + i for i in range(30)],
            'high': [105.0 + i for i in range(30)],
            'low': [95.0 + i for i in range(30)],
            'close': [100.0 + i for i in range(30)],
            'volume': [1000000] * 30,
            'amount': [100000000] * 30,
        })
        df.to_csv(Path(provider.data_dir) / '000001.SZ.csv', index=False)

        result = provider.get_bars(
            symbols=['000001.SZ'],
            start_date=datetime(2023, 1, 10),
            end_date=datetime(2023, 1, 20),
            timeframe='1d'
        )

        assert '000001.SZ' in result
        assert len(result['000001.SZ']) > 0

    def test_get_bars_no_file(self, provider):
        """Test getting bars when file doesn't exist."""
        provider.initialize()

        result = provider.get_bars(
            symbols=['NONEXISTENT.SZ'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10)
        )

        assert 'NONEXISTENT.SZ' not in result

    def test_get_latest_bar(self, provider):
        """Test getting latest bar."""
        provider.initialize()

        # Create sample data
        df = pd.DataFrame({
            'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
            'open': [100.0 + i for i in range(10)],
            'high': [105.0 + i for i in range(10)],
            'low': [95.0 + i for i in range(10)],
            'close': [100.0 + i for i in range(10)],
            'volume': [1000000] * 10,
            'amount': [100000000] * 10,
        })
        df.to_csv(Path(provider.data_dir) / '000001.SZ.csv', index=False)

        result = provider.get_latest_bar('000001.SZ')

        assert result is not None
        assert len(result) == 1

    def test_search_symbols(self, provider):
        """Test searching symbols."""
        provider.initialize()

        # Create sample files
        for symbol in ['000001.SZ', '000002.SZ', '600000.SH']:
            df = pd.DataFrame({
                'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='D'),
                'open': [100.0] * 10,
                'high': [105.0] * 10,
                'low': [95.0] * 10,
                'close': [100.0] * 10,
                'volume': [1000000] * 10,
                'amount': [100000000] * 10,
            })
            df.to_csv(Path(provider.data_dir) / f'{symbol}.csv', index=False)

        results = provider.search_symbols('000001')

        assert len(results) > 0
        assert any('000001' in r['symbol'] for r in results)


class TestTuShareProviderMock:
    """Test TuShareProvider with mocks."""

    @pytest.fixture
    def provider(self):
        """Create a TuShareProvider instance with mock token."""
        with patch.dict('os.environ', {'TUSHARE_TOKEN': 'mock_token'}):
            return TuShareProvider()

    def test_initialization(self, provider):
        """Test provider initialization."""
        with patch('tushare.pro_api') as mock_api:
            mock_pro = MagicMock()
            mock_pro.trade_cal.return_value = MagicMock(empty=False)
            mock_api.return_value = mock_pro

            provider.initialize()
            assert provider.pro_api is not None

    def test_get_supported_symbols_mock(self, provider):
        """Test getting symbols with mock."""
        with patch('tushare.pro_api') as mock_api:
            mock_pro = MagicMock()
            mock_df = MagicMock()
            mock_df.__getitem__.return_value = ['000001.SZ', '000002.SZ']
            mock_pro.stock_basic.return_value = mock_df
            mock_pro.trade_cal.return_value = MagicMock(empty=False)
            mock_api.return_value = mock_pro

            provider.initialize()
            symbols = provider.get_supported_symbols()

            assert len(symbols) == 2
            assert '000001.SZ' in symbols

    def test_get_bars_daily_mock(self, provider):
        """Test getting daily bars with mock."""
        with patch('tushare.pro_api') as mock_api:
            mock_pro = MagicMock()
            mock_df = MagicMock()
            mock_df.empty = False
            mock_df.rename.return_value = mock_df
            mock_df.__getitem__.return_value = mock_df
            mock_df.sort_values.return_value = mock_df
            mock_df.reset_index.return_value = mock_df
            mock_pro.daily.return_value = mock_df
            mock_pro.trade_cal.return_value = MagicMock(empty=False)
            mock_api.return_value = mock_pro

            provider.initialize()
            result = provider.get_bars(
                symbols=['000001.SZ'],
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 10),
                timeframe='1d'
            )

            assert '000001.SZ' in result


class TestAShareProvider:
    """Test cases for AShareProvider."""

    @pytest.fixture
    def provider(self):
        """Create an AShareProvider instance with mock credentials."""
        with patch.dict('os.environ', {
            'ASHARE_ENDPOINT': 'http://mock.endpoint',
            'ASHARE_API_KEY': 'mock_key'
        }):
            return AShareProvider()

    def test_initialization(self, provider):
        """Test provider initialization."""
        provider.initialize()
        assert provider._initialized is True

    def test_get_supported_symbols(self, provider):
        """Test getting supported symbols."""
        provider.initialize()
        symbols = provider.get_supported_symbols()

        assert isinstance(symbols, list)
        assert len(symbols) > 0
        assert '000001.SZ' in symbols

    def test_get_bars(self, provider):
        """Test getting bars."""
        provider.initialize()

        result = provider.get_bars(
            symbols=['000001.SZ'],
            start_date=datetime(2023, 1, 1),
            end_date=datetime(2023, 1, 10),
            timeframe='1d'
        )

        assert '000001.SZ' in result
        assert len(result['000001.SZ']) > 0

    def test_get_latest_bar(self, provider):
        """Test getting latest bar."""
        provider.initialize()

        result = provider.get_latest_bar('000001.SZ')

        assert result is not None
        assert len(result) == 1

    def test_search_symbols(self, provider):
        """Test searching symbols."""
        provider.initialize()

        results = provider.search_symbols('平安')

        assert isinstance(results, list)
        assert len(results) > 0

    def test_get_calendar(self, provider):
        """Test getting trading calendar."""
        provider.initialize()

        calendar = provider.get_calendar()

        assert isinstance(calendar, list)
        assert len(calendar) > 0


class TestProviderInterfaces:
    """Test that all providers implement the interface correctly."""

    def test_csv_provider_implements_interface(self):
        """Test CSVProvider implements IDataProvider."""
        from quanttool.domain.interfaces.data_provider import IDataProvider

        assert issubclass(CSVProvider, IDataProvider)

    def test_tushare_provider_implements_interface(self):
        """Test TuShareProvider implements IDataProvider."""
        from quanttool.domain.interfaces.data_provider import IDataProvider

        assert issubclass(TuShareProvider, IDataProvider)

    def test_ashare_provider_implements_interface(self):
        """Test AShareProvider implements IDataProvider."""
        from quanttool.domain.interfaces.data_provider import IDataProvider

        assert issubclass(AShareProvider, IDataProvider)
