"""Async Local Data Cache with PostgreSQL backend.

Key improvements over SQLite version:
1. No global locks - PostgreSQL MVCC handles concurrency
2. Connection pooling for efficient resource management
3. Row-level locking for atomic operations
4. Better support for distributed deployments
"""

import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd

from ..database.connection import get_connection_pool
from ..database.config import DatabaseConfig
from ...core.logging import get_logger

logger = get_logger(__name__)


class AsyncLocalDataCache:
    """
    Async local data cache with PostgreSQL backend.

    Features:
    - PostgreSQL metadata + Parquet data storage
    - TTL-based expiration
    - Row-level locking for concurrent access
    - Thread-safe operations without Python locks
    """

    def __init__(
        self,
        cache_dir: str = ".cache/stock_data",
        default_ttl: int = 86400,
        max_size_mb: int = 1024,
        config: Optional[DatabaseConfig] = None,
    ):
        """
        Initialize the async local cache.

        Args:
            cache_dir: Directory for cache storage
            default_ttl: Default TTL in seconds (default: 1 day)
            max_size_mb: Maximum cache size in MB
            config: Database configuration
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._config = config
        self._pool = None

    @property
    def pool(self):
        """Get the connection pool (lazy initialization)."""
        if self._pool is None:
            self._pool = get_connection_pool(self._config)
        return self._pool

    def _generate_key(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
    ) -> str:
        """Generate a unique cache key."""
        key_str = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Get cached data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            timeframe: Data timeframe

        Returns:
            DataFrame if cached and not expired, None otherwise
        """
        key = self._generate_key(symbol, start_date, end_date, timeframe)

        # Use transaction with row-level lock
        async with self.pool.transaction() as conn:
            row = await conn.fetchrow(
                """
                SELECT file_path, expires_at, row_count
                FROM cache_entries
                WHERE cache_key = $1
                FOR UPDATE
                """,
                key
            )

            if row is None:
                logger.debug(f"Cache miss for {symbol}")
                return None

            # Check expiration (use timezone-aware datetime)
            from datetime import timezone
            now = datetime.now(timezone.utc)
            expires_at = row["expires_at"]
            # Ensure both are timezone-aware for comparison
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)

            if now > expires_at:
                logger.debug(f"Cache expired for {symbol}")
                await self._delete_unlocked(conn, key)
                return None

            file_path = self.cache_dir / row["file_path"]
            if not file_path.exists():
                logger.warning(f"Cache file missing for {symbol}: {row['file_path']}")
                await self._delete_unlocked(conn, key)
                return None

        # Read parquet outside transaction
        try:
            df = pd.read_parquet(file_path)
            logger.debug(f"Cache hit for {symbol}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to read cache for {symbol}: {e}")
            return None

    async def set(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data: pd.DataFrame,
        timeframe: str = "1d",
        ttl: Optional[int] = None,
    ) -> bool:
        """
        Cache data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data: DataFrame to cache
            timeframe: Data timeframe
            ttl: TTL in seconds (uses default if None)

        Returns:
            True if cached successfully, False otherwise
        """
        if data is None or data.empty:
            logger.warning(f"Attempted to cache empty data for {symbol}")
            return False

        key = self._generate_key(symbol, start_date, end_date, timeframe)
        ttl = ttl or self.default_ttl

        # Convert string dates to date objects for PostgreSQL
        if isinstance(start_date, str):
            from datetime import datetime as dt
            start_date = dt.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            from datetime import datetime as dt
            end_date = dt.strptime(end_date, "%Y-%m-%d").date()

        # Generate file path
        file_path = f"{key}.parquet"
        full_path = self.cache_dir / file_path

        try:
            # Clean data (remove unhashable types)
            data_to_cache = data.copy()
            cols_to_drop = []

            for col in data_to_cache.columns:
                try:
                    sample = data_to_cache[col].dropna()
                    if len(sample) > 0:
                        for val in sample.values:
                            if isinstance(val, (dict, list, set)):
                                cols_to_drop.append(col)
                                logger.debug(f"Dropping unhashable column '{col}' for {symbol}")
                                break
                except Exception:
                    cols_to_drop.append(col)

            if cols_to_drop:
                data_to_cache = data_to_cache.drop(columns=cols_to_drop)

            if data_to_cache.empty or len(data_to_cache.columns) == 0:
                logger.warning(f"No hashable columns left for {symbol}, skipping cache")
                return False

            # Save to parquet
            data_to_cache.to_parquet(full_path, compression='snappy', index=False)

            size_bytes = full_path.stat().st_size

            # Calculate data hash
            try:
                data_hash = hashlib.md5(
                    pd.util.hash_pandas_object(data_to_cache).values.tobytes()
                ).hexdigest()
            except Exception:
                data_hash = None

            # Update metadata
            now = datetime.now()
            expires_at = now + timedelta(seconds=ttl)

            async with self.pool.transaction() as conn:
                await conn.execute(
                    """
                    INSERT INTO cache_entries
                    (cache_key, file_path, created_at, expires_at, data_hash, row_count, size_bytes,
                     symbol, start_date, end_date, timeframe)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                    ON CONFLICT (cache_key) DO UPDATE SET
                        file_path = EXCLUDED.file_path,
                        created_at = EXCLUDED.created_at,
                        expires_at = EXCLUDED.expires_at,
                        data_hash = EXCLUDED.data_hash,
                        row_count = EXCLUDED.row_count,
                        size_bytes = EXCLUDED.size_bytes,
                        symbol = EXCLUDED.symbol,
                        start_date = EXCLUDED.start_date,
                        end_date = EXCLUDED.end_date,
                        timeframe = EXCLUDED.timeframe
                    """,
                    key, file_path, now, expires_at, data_hash,
                    len(data_to_cache), size_bytes,
                    symbol, start_date, end_date, timeframe
                )

            logger.debug(f"Cached {len(data_to_cache)} rows for {symbol} ({size_bytes} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to cache data for {symbol}: {e}")
            return False

    async def delete(self, key: str) -> bool:
        """Delete a cache entry by key."""
        async with self.pool.transaction() as conn:
            return await self._delete_unlocked(conn, key)

    async def _delete_unlocked(self, conn, key: str) -> bool:
        """Delete a cache entry (assumes transaction is active)."""
        try:
            row = await conn.fetchrow(
                "SELECT file_path FROM cache_entries WHERE cache_key = $1",
                key
            )

            if row:
                file_path = self.cache_dir / row["file_path"]
                if file_path.exists():
                    file_path.unlink()

                await conn.execute(
                    "DELETE FROM cache_entries WHERE cache_key = $1",
                    key
                )

            return True
        except Exception as e:
            logger.error(f"Failed to delete cache entry {key}: {e}")
            return False

    async def clear_expired(self) -> int:
        """Clear all expired cache entries."""
        rows = await self.pool.fetch(
            """
            SELECT cache_key, file_path FROM cache_entries
            WHERE expires_at < NOW()
            """
        )

        count = 0
        async with self.pool.transaction() as conn:
            for row in rows:
                file_path = self.cache_dir / row["file_path"]
                if file_path.exists():
                    file_path.unlink()

                await conn.execute(
                    "DELETE FROM cache_entries WHERE cache_key = $1",
                    row["cache_key"]
                )
                count += 1

        if count > 0:
            logger.debug(f"Cleared {count} expired cache entries")

        return count

    async def clear_all(self) -> int:
        """Clear all cache entries."""
        rows = await self.pool.fetch("SELECT file_path FROM cache_entries")

        count = 0
        async with self.pool.transaction() as conn:
            for row in rows:
                file_path = self.cache_dir / row["file_path"]
                if file_path.exists():
                    file_path.unlink()
                    count += 1

            await conn.execute("DELETE FROM cache_entries")

        logger.debug(f"Cleared all cache: {count} files")
        return count

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        row = await self.pool.fetchrow(
            """
            SELECT
                COUNT(*) as entry_count,
                COALESCE(SUM(row_count), 0) as total_rows,
                COALESCE(SUM(size_bytes), 0) as total_size_bytes,
                COUNT(CASE WHEN expires_at > NOW() THEN 1 END) as active_entries,
                COUNT(CASE WHEN expires_at <= NOW() THEN 1 END) as expired_entries
            FROM cache_entries
        """
        )

        return {
            "entry_count": row["entry_count"],
            "total_rows": row["total_rows"],
            "total_size_bytes": row["total_size_bytes"],
            "total_size_mb": round(row["total_size_bytes"] / (1024 * 1024), 2),
            "active_entries": row["active_entries"],
            "expired_entries": row["expired_entries"],
            "cache_dir": str(self.cache_dir),
        }

    async def list_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List cache entries."""
        rows = await self.pool.fetch(
            """
            SELECT cache_key, file_path, created_at, expires_at, row_count, size_bytes, symbol
            FROM cache_entries
            ORDER BY created_at DESC
            LIMIT $1
            """,
            limit
        )

        return [dict(row) for row in rows]

    async def get_entries_by_symbol(self, symbol: str) -> List[Dict[str, Any]]:
        """Get all cache entries for a symbol."""
        rows = await self.pool.fetch(
            """
            SELECT cache_key, file_path, created_at, expires_at, row_count, size_bytes,
                   start_date, end_date, timeframe
            FROM cache_entries
            WHERE symbol = $1
            ORDER BY created_at DESC
            """,
            symbol
        )

        return [dict(row) for row in rows]

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


# Singleton instance
_instance: Optional[AsyncLocalDataCache] = None


def get_async_local_cache(config: Optional[DatabaseConfig] = None) -> AsyncLocalDataCache:
    """Get the async local cache singleton."""
    global _instance
    if _instance is None:
        _instance = AsyncLocalDataCache(config=config)
    return _instance
