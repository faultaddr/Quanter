"""Unit tests for LocalDataCache."""

import os
import tempfile
import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from quanttool.infrastructure.cache import LocalDataCache


class TestLocalDataCache:
    """Tests for LocalDataCache functionality."""

    @pytest.fixture
    def cache(self):
        """Create a temporary cache for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "test_cache")
            yield LocalDataCache(cache_dir=cache_dir, default_ttl=3600)

    @pytest.fixture
    def sample_data(self):
        """Create sample OHLCV data."""
        return pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'open': [10.0] * 10,
            'high': [11.0] * 10,
            'low': [9.0] * 10,
            'close': [10.5] * 10,
            'volume': [1000000] * 10,
            'amount': [10500000] * 10
        })

    def test_cache_set_and_get(self, cache, sample_data):
        """Test basic set and get operations."""
        symbol = "000001.SZ"
        start_date = "2024-01-01"
        end_date = "2024-01-10"

        # Set data
        result = cache.set(symbol, start_date, end_date, sample_data)
        assert result is True

        # Get data
        cached = cache.get(symbol, start_date, end_date)
        assert cached is not None
        assert len(cached) == len(sample_data)
        assert list(cached.columns) == list(sample_data.columns)

    def test_cache_miss(self, cache):
        """Test cache miss returns None."""
        result = cache.get("999999.SZ", "2024-01-01", "2024-01-10")
        assert result is None

    def test_cache_expiration(self, sample_data):
        """Test that expired cache entries are not returned."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "test_cache")
            # Create cache with very short TTL
            cache = LocalDataCache(cache_dir=cache_dir, default_ttl=1)

            symbol = "000001.SZ"
            start_date = "2024-01-01"
            end_date = "2024-01-10"

            cache.set(symbol, start_date, end_date, sample_data)

            # Should be present immediately
            cached = cache.get(symbol, start_date, end_date)
            assert cached is not None

            # Wait for expiration
            import time
            time.sleep(2)

            # Should be expired now
            cached = cache.get(symbol, start_date, end_date)
            assert cached is None

    def test_cache_clear_expired(self, cache, sample_data):
        """Test clearing expired entries."""
        # Add multiple entries
        for i in range(3):
            symbol = f"00000{i}.SZ"
            cache.set(symbol, "2024-01-01", "2024-01-10", sample_data)

        stats = cache.get_stats()
        assert stats["entry_count"] == 3

        # Clear expired (none should be expired yet)
        count = cache.clear_expired()
        assert count == 0

        stats = cache.get_stats()
        assert stats["entry_count"] == 3

    def test_cache_clear_all(self, cache, sample_data):
        """Test clearing all entries."""
        # Add multiple entries
        for i in range(5):
            symbol = f"00000{i}.SZ"
            cache.set(symbol, "2024-01-01", "2024-01-10", sample_data)

        stats = cache.get_stats()
        assert stats["entry_count"] == 5

        # Clear all
        count = cache.clear_all()
        assert count == 5

        stats = cache.get_stats()
        assert stats["entry_count"] == 0

    def test_cache_stats(self, cache, sample_data):
        """Test cache statistics."""
        # Add entries
        for i in range(3):
            symbol = f"00000{i}.SZ"
            cache.set(symbol, "2024-01-01", "2024-01-10", sample_data)

        stats = cache.get_stats()

        assert stats["entry_count"] == 3
        assert stats["total_rows"] == len(sample_data) * 3
        assert stats["total_size_bytes"] > 0
        assert stats["total_size_mb"] > 0

    def test_cache_list_entries(self, cache, sample_data):
        """Test listing cache entries."""
        # Add entries
        symbols = ["000001.SZ", "000002.SZ", "000003.SZ"]
        for symbol in symbols:
            cache.set(symbol, "2024-01-01", "2024-01-10", sample_data)

        entries = cache.list_entries()
        assert len(entries) == 3

        for entry in entries:
            assert "key" in entry
            assert "file" in entry
            assert "created" in entry
            assert "expires" in entry
            assert "rows" in entry

    def test_cache_overwrite(self, cache, sample_data):
        """Test that setting same key overwrites existing data."""
        symbol = "000001.SZ"
        start_date = "2024-01-01"
        end_date = "2024-01-10"

        # Set initial data
        cache.set(symbol, start_date, end_date, sample_data)

        # Set new data with different values
        new_data = sample_data.copy()
        new_data['close'] = [20.0] * len(new_data)
        cache.set(symbol, start_date, end_date, new_data)

        # Should have new data
        cached = cache.get(symbol, start_date, end_date)
        assert cached is not None
        assert all(cached['close'] == 20.0)

    def test_cache_empty_data(self, cache):
        """Test that empty data is not cached."""
        empty_df = pd.DataFrame()
        result = cache.set("000001.SZ", "2024-01-01", "2024-01-10", empty_df)
        assert result is False

    def test_cache_different_timeframes(self, cache, sample_data):
        """Test caching different timeframes separately."""
        symbol = "000001.SZ"
        start_date = "2024-01-01"
        end_date = "2024-01-10"

        # Cache daily data
        cache.set(symbol, start_date, end_date, sample_data, timeframe="1d")

        # Cache weekly data
        weekly_data = sample_data.head(2)
        cache.set(symbol, start_date, end_date, weekly_data, timeframe="1w")

        # Should get different data for different timeframes
        daily = cache.get(symbol, start_date, end_date, timeframe="1d")
        weekly = cache.get(symbol, start_date, end_date, timeframe="1w")

        assert len(daily) == 10
        assert len(weekly) == 2

    def test_cache_context_manager(self, sample_data):
        """Test using cache as context manager."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_dir = os.path.join(tmpdir, "test_cache")

            with LocalDataCache(cache_dir=cache_dir) as cache:
                cache.set("000001.SZ", "2024-01-01", "2024-01-10", sample_data)
                cached = cache.get("000001.SZ", "2024-01-01", "2024-01-10")
                assert cached is not None