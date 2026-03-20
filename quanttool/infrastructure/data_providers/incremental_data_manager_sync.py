"""Synchronous wrapper for AsyncIncrementalDataManager.

This module provides a synchronous interface for the async IncrementalDataManager,
allowing existing synchronous code to use PostgreSQL without modification.
"""

from datetime import datetime
from typing import Optional, Dict, Any, List
import pandas as pd

from .incremental_data_manager_async import (
    AsyncIncrementalDataManager,
    DataType,
    DataRange,
    get_async_incremental_manager
)
from ..database.config import DatabaseConfig
from ..database.sync_loop import run_async


class IncrementalDataManager:
    """
    Synchronous wrapper for AsyncIncrementalDataManager.

    This class provides the same interface as the original SQLite-based version,
    but uses PostgreSQL internally.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/incremental_data",
        default_ttl_days: int = 1,
        max_cache_size_mb: int = 2048,
        config: Optional[DatabaseConfig] = None,
    ):
        """Initialize the synchronous wrapper."""
        self.cache_dir = cache_dir
        self.default_ttl_days = default_ttl_days
        self.max_cache_bytes = max_cache_size_mb * 1024 * 1024
        self._config = config
        self._async_manager = None

    @property
    def async_manager(self) -> AsyncIncrementalDataManager:
        """Get the async manager instance."""
        if self._async_manager is None:
            self._async_manager = AsyncIncrementalDataManager(
                cache_dir=self.cache_dir,
                default_ttl_days=self.default_ttl_days,
                max_cache_size_mb=self.max_cache_bytes // (1024 * 1024),
                config=self._config,
            )
        return self._async_manager

    def get_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        fetcher,
        data_type: str = DataType.STOCK_BAR,
        force_refresh: bool = False,
        skip_network_fetch: bool = False,
    ) -> pd.DataFrame:
        """Get data with automatic incremental fetching."""
        return run_async(
            self.async_manager.get_data(symbol, start_date, end_date, fetcher, data_type, force_refresh, skip_network_fetch)
        )

    def get_cache_stats(self, data_type: str = None) -> Dict[str, Any]:
        """Get cache statistics."""
        return run_async(self.async_manager.get_cache_stats(data_type))

    def list_symbols(self, data_type: str = None) -> List[Dict[str, Any]]:
        """List all cached symbols."""
        return run_async(self.async_manager.list_symbols(data_type))

    def clear_expired(self) -> int:
        """Clear expired data."""
        return run_async(self.async_manager.clear_expired())

    def update_latest(
        self,
        symbols: List[str],
        fetcher,
        days_back: int = 30,
        data_type: str = DataType.STOCK_BAR,
    ) -> Dict[str, int]:
        """Batch update latest data."""
        return run_async(
            self.async_manager.update_latest(symbols, fetcher, days_back, data_type)
        )

    # Convenience methods
    def get_stock_data(
        self,
        symbol: str,
        days: int = 120,
        fetcher=None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Get stock K-line data."""
        return run_async(
            self.async_manager.get_stock_data(symbol, days, fetcher, force_refresh)
        )

    def get_index_data(
        self,
        index_code: str,
        days: int = 120,
        fetcher=None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Get index K-line data."""
        return run_async(
            self.async_manager.get_index_data(index_code, days, fetcher, force_refresh)
        )

    def get_money_flow(
        self,
        symbol: str,
        days: int = 60,
        fetcher=None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Get money flow data."""
        return run_async(
            self.async_manager.get_money_flow(symbol, days, fetcher, force_refresh)
        )

    def close(self) -> None:
        """Close the manager."""
        if self._async_manager:
            run_async(self._async_manager.close())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def get_data_range(
        self,
        symbol: str,
        data_type: str = DataType.STOCK_BAR,
    ) -> Optional[DataRange]:
        """Get cached data range information (read-only, no lock)."""
        return run_async(self.async_manager.get_data_range(symbol, data_type))

    # Alias for backward compatibility
    def _get_data_range(
        self,
        symbol: str,
        data_type: str = DataType.STOCK_BAR,
    ) -> Optional[DataRange]:
        """Get cached data range information (backward compatibility alias)."""
        return self.get_data_range(symbol, data_type)

    # Backward compatibility
    def _init_db(self):
        """Initialize database (handled by async version)."""
        pass

    def _migrate_old_schema(self):
        """Migrate old schema (handled by async version)."""
        pass

    def _has_trading_days(self, start_date, end_date) -> bool:
        """Check for trading days."""
        from datetime import timedelta
        if start_date > end_date:
            return False
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                return True
            current += timedelta(days=1)
        return False


# Singleton instance
_instance: Optional[IncrementalDataManager] = None


def get_incremental_manager() -> IncrementalDataManager:
    """Get the incremental manager singleton."""
    global _instance
    if _instance is None:
        _instance = IncrementalDataManager()
    return _instance
