"""Synchronous wrapper for AsyncLocalDataCache.

This module provides a synchronous interface for the async LocalDataCache,
allowing existing synchronous code to use PostgreSQL without modification.
"""

import asyncio
import concurrent.futures
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from .local_cache_async import AsyncLocalDataCache, get_async_local_cache
from ..database.config import DatabaseConfig


# Global event loop for synchronous operations
_loop = None
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)


def _get_loop():
    """Get or create the global event loop."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


class LocalDataCache:
    """
    Synchronous wrapper for AsyncLocalDataCache.

    This class provides the same interface as the original SQLite-based version,
    but uses PostgreSQL internally.
    """

    def __init__(
        self,
        cache_dir: str = ".cache/stock_data",
        default_ttl: int = 86400,
        max_size_mb: int = 1024,
        config: Optional[DatabaseConfig] = None,
    ):
        """Initialize the synchronous wrapper."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._config = config
        self._async_cache = None

    @property
    def async_cache(self) -> AsyncLocalDataCache:
        """Get the async cache instance."""
        if self._async_cache is None:
            self._async_cache = AsyncLocalDataCache(
                cache_dir=str(self.cache_dir),
                default_ttl=self.default_ttl,
                max_size_mb=self.max_size_bytes // (1024 * 1024),
                config=self._config,
            )
        return self._async_cache

    def _run_async(self, coro):
        """Run an async coroutine synchronously using a dedicated thread."""
        def run_in_thread():
            loop = _get_loop()
            asyncio.set_event_loop(loop)
            return loop.run_until_complete(coro)

        try:
            loop = asyncio.get_running_loop()
            future = _executor.submit(run_in_thread)
            return future.result()
        except RuntimeError:
            loop = _get_loop()
            return loop.run_until_complete(coro)

    def get(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """Get cached data for a symbol."""
        return self._run_async(
            self.async_cache.get(symbol, start_date, end_date, timeframe)
        )

    def set(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data: pd.DataFrame,
        timeframe: str = "1d",
        ttl: Optional[int] = None,
    ) -> bool:
        """Cache data for a symbol."""
        return self._run_async(
            self.async_cache.set(symbol, start_date, end_date, data, timeframe, ttl)
        )

    def delete(self, key: str) -> bool:
        """Delete a cache entry by key."""
        return self._run_async(self.async_cache.delete(key))

    def clear_expired(self) -> int:
        """Clear all expired cache entries."""
        return self._run_async(self.async_cache.clear_expired())

    def clear_all(self) -> int:
        """Clear all cache entries."""
        return self._run_async(self.async_cache.clear_all())

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return self._run_async(self.async_cache.get_stats())

    def list_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List cache entries."""
        return self._run_async(self.async_cache.list_entries(limit))

    def close(self) -> None:
        """Close the cache."""
        if self._async_cache:
            self._run_async(self._async_cache.close())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # Backward compatibility
    def _init_db(self):
        """Initialize database (handled by async version)."""
        pass

    def _generate_key(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
    ) -> str:
        """Generate a unique cache key."""
        import hashlib
        key_str = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        return hashlib.md5(key_str.encode()).hexdigest()


# Singleton instance
_instance: Optional[LocalDataCache] = None


def get_local_cache() -> LocalDataCache:
    """Get the local cache singleton."""
    global _instance
    if _instance is None:
        _instance = LocalDataCache()
    return _instance
