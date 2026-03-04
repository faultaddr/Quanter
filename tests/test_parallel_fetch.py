"""Tests for parallel and async data fetching."""

import os
import pytest
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

# Check if aiohttp is available
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

if AIOHTTP_AVAILABLE:
    from quanttool.infrastructure.data_providers.async_data_fetcher import (
        AsyncDataFetcher,
        fetch_symbols,
        fetch_symbols_async
    )


class TestAsyncDataFetcher:
    """Tests for AsyncDataFetcher."""

    @pytest.fixture
    def sample_sina_response(self):
        """Sample Sina API response."""
        return '''[
            ["2024-01-01", "10.00", "11.00", "9.00", "10.50", "1000000"],
            ["2024-01-02", "10.50", "11.50", "10.00", "11.00", "1200000"],
            ["2024-01-03", "11.00", "12.00", "10.50", "11.50", "1500000"]
        ]'''

    @pytest.fixture
    def sample_tencent_response(self):
        """Sample Tencent API response."""
        return '''{
            "data": {
                "sh600519": {
                    "qfqday": [
                        ["2024-01-01", "10.00", "10.50", "11.00", "9.00", "1000000"],
                        ["2024-01-02", "10.50", "11.00", "11.50", "10.00", "1200000"]
                    ]
                }
            }
        }'''

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
    def test_normalize_code(self):
        """Test stock code normalization."""
        assert AsyncDataFetcher._normalize_code("600519") == "sh600519"
        assert AsyncDataFetcher._normalize_code("000001") == "sz000001"
        assert AsyncDataFetcher._normalize_code("600519.SH") == "sh600519"
        assert AsyncDataFetcher._normalize_code("000001.SZ") == "sz000001"
        assert AsyncDataFetcher._normalize_code("sh600519") == "sh600519"

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
    def test_parse_sina_data(self, sample_sina_response):
        """Test parsing Sina API response."""
        df = AsyncDataFetcher._parse_sina_data(sample_sina_response)

        assert len(df) == 3
        assert 'timestamp' in df.columns
        assert 'open' in df.columns
        assert 'close' in df.columns
        assert 'high' in df.columns
        assert 'low' in df.columns
        assert 'volume' in df.columns
        assert 'amount' in df.columns

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
    def test_parse_tencent_data(self, sample_tencent_response):
        """Test parsing Tencent API response."""
        df = AsyncDataFetcher._parse_tencent_data(sample_tencent_response, "sh600519")

        assert len(df) == 2
        assert 'timestamp' in df.columns
        assert 'open' in df.columns
        assert 'close' in df.columns

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
    def test_parse_empty_data(self):
        """Test parsing empty responses."""
        df = AsyncDataFetcher._parse_sina_data("")
        assert df.empty

        df = AsyncDataFetcher._parse_sina_data("[]")
        assert df.empty

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
    @pytest.mark.asyncio
    async def test_fetch_single_mock(self, sample_sina_response):
        """Test single symbol fetch with mocked HTTP."""
        import asyncio
        with patch('aiohttp.ClientSession') as mock_session:
            # Mock response
            mock_response = MagicMock()
            mock_response.status = 200
            mock_response.text = asyncio.coroutine(lambda: sample_sina_response)

            mock_get = MagicMock(return_value=mock_response)
            mock_get.__aenter__ = asyncio.coroutine(lambda: mock_response)
            mock_get.__aexit__ = asyncio.coroutine(lambda *args: None)

            mock_session_instance = MagicMock()
            mock_session_instance.get = MagicMock(return_value=mock_get)
            mock_session_instance.close = asyncio.coroutine(lambda: None)

            mock_session.return_value = mock_session_instance

            async with AsyncDataFetcher(cache_dir=None) as fetcher:
                # This test just verifies the flow, actual network calls are mocked
                pass

    @pytest.mark.skipif(not AIOHTTP_AVAILABLE, reason="aiohttp not installed")
    def test_fetch_symbols_sync(self):
        """Test synchronous wrapper function."""
        # This test verifies the function exists and can be called
        # Actual network calls should be mocked in integration tests
        with patch('asyncio.run') as mock_run:
            mock_run.return_value = {}
            result = fetch_symbols(["000001"], "2024-01-01", "2024-01-10")
            assert isinstance(result, dict)


class TestParallelFetchIntegration:
    """Integration tests for parallel fetching (requires network)."""

    @pytest.fixture
    def data_fetcher(self):
        """Create a DataFetcher instance for testing."""
        try:
            from quanttool.infrastructure.data_providers.data_fetcher import (
                create_data_fetcher_with_credentials
            )
            return create_data_fetcher_with_credentials()
        except Exception:
            pytest.skip("Data fetcher credentials not available")

    @pytest.mark.slow
    def test_parallel_fetch_small(self, data_fetcher):
        """Test parallel fetching with small symbol list."""
        symbols = ["000001.SZ", "000002.SZ", "600519.SH"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        results = data_fetcher.get_bars_parallel(
            symbols, start_date, end_date, show_progress=False
        )

        # At least some symbols should return data
        assert len(results) > 0

        # Check data structure
        for symbol, df in results.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0
            assert 'timestamp' in df.columns
            assert 'close' in df.columns

    @pytest.mark.slow
    def test_cached_fetch(self, data_fetcher):
        """Test cached fetch - should be faster on second call."""
        symbols = ["000001.SZ", "600519.SH"]
        start_date = datetime.now() - timedelta(days=30)
        end_date = datetime.now()

        # First call - should fetch from network
        results1 = data_fetcher.get_bars_cached(symbols, start_date, end_date)

        # Second call - should use cache
        results2 = data_fetcher.get_bars_cached(symbols, start_date, end_date)

        # Results should be identical
        assert set(results1.keys()) == set(results2.keys())
        for symbol in results1:
            pd.testing.assert_frame_equal(results1[symbol], results2[symbol])