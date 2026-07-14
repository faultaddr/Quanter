"""
Async Incremental Data Manager with PostgreSQL backend.

Key improvements over SQLite version:
1. No global locks - uses PostgreSQL transactions with row-level locking
2. True concurrency - MVCC allows parallel reads and writes
3. Connection pooling - efficient resource management
4. Better error handling and recovery
"""

import os
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass

from ...database.connection import get_connection_pool
from ...database.config import DatabaseConfig
from ....core.logging import get_logger

logger = get_logger(__name__)


class DataType:
    """Data type constants"""
    STOCK_BAR = "stock_bar"
    INDEX_BAR = "index_bar"
    MONEY_FLOW = "money_flow"
    FINANCE = "finance"


@dataclass
class DataRange:
    """Data range information"""
    symbol: str
    data_type: str = DataType.STOCK_BAR
    earliest_date: datetime = None
    latest_date: datetime = None
    row_count: int = 0
    last_updated: datetime = None


class AsyncIncrementalDataManager:
    """
    Async Incremental Data Manager with PostgreSQL backend.

    Core features:
    1. Per-symbol data storage
    2. Smart fetch range calculation
    3. Automatic data merging
    4. Row-level locking for concurrent updates
    5. TTL-based expiration

    Usage:
        manager = AsyncIncrementalDataManager()
        df = await manager.get_data("000001.SZ", start_date, end_date, fetcher)
    """

    def __init__(
        self,
        cache_dir: str = ".cache/incremental_data",
        default_ttl_days: int = 1,
        max_cache_size_mb: int = 2048,
        config: Optional[DatabaseConfig] = None,
    ):
        """
        Initialize the async incremental data manager.

        Args:
            cache_dir: Cache directory for parquet files
            default_ttl_days: Data expiration days
            max_cache_size_mb: Maximum cache size
            config: Database configuration
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl_days = default_ttl_days
        self.max_cache_bytes = max_cache_size_mb * 1024 * 1024
        self._config = config
        self._pool = None

    @property
    def pool(self):
        """Get the connection pool (lazy initialization)."""
        if self._pool is None:
            self._pool = get_connection_pool(self._config)
        return self._pool

    async def get_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        fetcher,
        data_type: str = DataType.STOCK_BAR,
        force_refresh: bool = False,
        skip_network_fetch: bool = False,
    ) -> pd.DataFrame:
        """
        Get data with automatic incremental fetching.

        Args:
            symbol: Stock/index code
            start_date: Start date
            end_date: End date
            fetcher: Data fetcher with get_bars method
            data_type: Data type
            force_refresh: Force refresh cache
            skip_network_fetch: Skip network fetching, only return cached data

        Returns:
            DataFrame with requested data
        """
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Phase 1: Check cache with row-level lock
        async with self.pool.transaction() as conn:
            cached_range = await self._get_data_range_locked(conn, symbol, data_type)

            if force_refresh and cached_range:
                logger.debug(f"[{symbol}][{data_type}] Force refresh, deleting cache")
                await self._delete_data(conn, symbol, data_type)
                cached_range = None

            fetch_ranges = self._calculate_fetch_ranges(
                symbol, start_date, end_date, cached_range
            )

            # Skip network fetch if requested
            if skip_network_fetch:
                fetch_ranges = []

            # Read cached data within transaction
            cached_data = None
            if cached_range and not force_refresh:
                cached_data = await self._load_data(symbol, data_type)

        # Phase 2: Network requests (no lock needed, can run in parallel)
        all_new_data = []
        for fetch_start, fetch_end in fetch_ranges:
            logger.debug(f"[{symbol}][{data_type}] Fetching: {fetch_start} ~ {fetch_end}")
            try:
                new_data = fetcher.get_bars([symbol], fetch_start, fetch_end, "1d")
                if symbol in new_data and not new_data[symbol].empty:
                    all_new_data.append(new_data[symbol])
            except Exception as e:
                logger.error(f"[{symbol}][{data_type}] Fetch failed: {e}")

        # Phase 3: Merge and save (with transaction)
        if cached_data is not None and not cached_data.empty:
            all_new_data.insert(0, cached_data)

        if all_new_data:
            final_data = pd.concat(all_new_data, ignore_index=True)

            # Find date column
            date_col = None
            for col in ['timestamp', 'trade_date', 'date']:
                if col in final_data.columns:
                    date_col = col
                    break

            if date_col:
                final_data[date_col] = pd.to_datetime(final_data[date_col])
                final_data = final_data.drop_duplicates(subset=[date_col], keep='last')
                final_data = final_data.sort_values(date_col).reset_index(drop=True)

            # Save to cache
            if not final_data.empty:
                async with self.pool.transaction() as conn:
                    await self._save_data(conn, symbol, final_data, data_type)

            # Filter to requested range
            if date_col:
                final_data = final_data[
                    (final_data[date_col] >= start_date) &
                    (final_data[date_col] <= end_date)
                ]

            return final_data

        return pd.DataFrame()

    async def get_data_range(
        self, symbol: str, data_type: str = DataType.STOCK_BAR
    ) -> Optional[DataRange]:
        """Get cached data range information (read-only, no lock)."""
        row = await self.pool.fetchrow(
            """
            SELECT symbol, data_type, earliest_date, latest_date, row_count, last_updated
            FROM data_ranges
            WHERE symbol = $1 AND data_type = $2
            """,
            symbol, data_type
        )

        if row is None:
            return None

        return DataRange(
            symbol=row["symbol"],
            data_type=row["data_type"],
            earliest_date=row["earliest_date"],
            latest_date=row["latest_date"],
            row_count=row["row_count"],
            last_updated=row["last_updated"],
        )

    async def _get_data_range_locked(
        self, conn, symbol: str, data_type: str
    ) -> Optional[DataRange]:
        """Get data range with row-level lock (SELECT FOR UPDATE)."""
        row = await conn.fetchrow(
            """
            SELECT symbol, data_type, earliest_date, latest_date, row_count, last_updated
            FROM data_ranges
            WHERE symbol = $1 AND data_type = $2
            FOR UPDATE
            """,
            symbol, data_type
        )

        if row is None:
            return None

        return DataRange(
            symbol=row["symbol"],
            data_type=row["data_type"],
            earliest_date=row["earliest_date"],
            latest_date=row["latest_date"],
            row_count=row["row_count"],
            last_updated=row["last_updated"],
        )

    def _calculate_fetch_ranges(
        self,
        symbol: str,
        start_date: datetime,
        end_date: datetime,
        cached_range: Optional[DataRange],
        min_coverage_threshold: float = 0.95,
    ) -> List[Tuple[datetime, datetime]]:
        """Calculate date ranges that need to be fetched."""
        fetch_ranges = []

        today = datetime.now().date()
        yesterday = today - timedelta(days=1)
        start_date_day = start_date.date() if hasattr(start_date, 'date') else start_date
        end_date_day = end_date.date() if hasattr(end_date, 'date') else end_date

        # Case 1: No cache
        if cached_range is None:
            logger.debug(f"[{symbol}] No cache, fetching full range: {start_date_day} ~ {end_date_day}")
            return [(start_date, end_date)]

        earliest = cached_range.earliest_date
        latest = cached_range.latest_date
        if hasattr(earliest, 'date'):
            earliest = earliest.date()
        if hasattr(latest, 'date'):
            latest = latest.date()

        # Calculate coverage
        request_days = (end_date_day - start_date_day).days + 1
        overlap_start = max(start_date_day, earliest)
        overlap_end = min(end_date_day, latest)
        if overlap_start <= overlap_end:
            covered_days = (overlap_end - overlap_start).days + 1
            coverage = covered_days / request_days if request_days > 0 else 1.0
        else:
            coverage = 0.0

        # Skip if coverage is high enough
        if coverage >= min_coverage_threshold:
            logger.debug(f"[{symbol}] Cache coverage {coverage:.1%} >= {min_coverage_threshold:.0%}, skipping fetch")
            return []

        # Need front data?
        if start_date_day < earliest:
            fetch_start = start_date
            fetch_end = datetime.combine(earliest - timedelta(days=1), datetime.max.time())
            fetch_end = min(fetch_end, end_date)

            if self._has_trading_days(fetch_start.date(), fetch_end.date()):
                logger.debug(f"[{symbol}] Fetching front: {fetch_start.date()} ~ {fetch_end.date()}")
                fetch_ranges.append((fetch_start, fetch_end))

        # Need back data?
        if latest < yesterday:
            effective_end = min(end_date_day, yesterday)
            if effective_end > latest:
                fetch_start = datetime.combine(latest + timedelta(days=1), datetime.min.time())
                fetch_start = max(fetch_start, start_date)
                fetch_end = datetime.combine(effective_end, datetime.max.time())
                fetch_end = min(fetch_end, end_date)

                if self._has_trading_days(fetch_start.date(), fetch_end.date()):
                    logger.debug(f"[{symbol}] Fetching back: {fetch_start.date()} ~ {fetch_end.date()}")
                    fetch_ranges.append((fetch_start, fetch_end))
        elif latest >= yesterday:
            logger.debug(f"[{symbol}] Cache latest {latest} >= yesterday {yesterday}, data is current")

        if not fetch_ranges:
            logger.debug(f"[{symbol}] Cache fully covers ({earliest} ~ {latest}), no fetch needed")

        return fetch_ranges

    def _has_trading_days(self, start_date, end_date) -> bool:
        """Check if date range might contain trading days."""
        if start_date > end_date:
            return False

        current = start_date
        while current <= end_date:
            if current.weekday() < 5:  # Monday to Friday
                return True
            current += timedelta(days=1)

        return False

    async def _load_data(
        self, symbol: str, data_type: str = DataType.STOCK_BAR
    ) -> Optional[pd.DataFrame]:
        """Load cached data from parquet file."""
        row = await self.pool.fetchrow(
            "SELECT file_path FROM data_ranges WHERE symbol = $1 AND data_type = $2",
            symbol, data_type
        )

        if row is None:
            return None

        file_path = self.cache_dir / row["file_path"]
        if not file_path.exists():
            logger.warning(f"[{symbol}][{data_type}] Cache file not found: {file_path}")
            return None

        try:
            return pd.read_parquet(file_path)
        except Exception as e:
            logger.error(f"[{symbol}][{data_type}] Failed to read cache: {e}")
            return None

    async def _save_data(
        self, conn, symbol: str, data: pd.DataFrame, data_type: str = DataType.STOCK_BAR
    ) -> bool:
        """Save data to cache."""
        if data.empty:
            return False

        # Find date column
        date_col = None
        for col in ['timestamp', 'trade_date', 'date']:
            if col in data.columns:
                date_col = col
                break

        if date_col is None:
            logger.error(f"[{symbol}][{data_type}] No date column found")
            return False

        save_data = data.copy()
        save_data[date_col] = pd.to_datetime(save_data[date_col])

        # Clean data (remove dict/list values)
        rows_to_drop = set()
        for col in save_data.columns:
            if col == date_col:
                continue
            for idx, val in save_data[col].items():
                if isinstance(val, (dict, list)):
                    rows_to_drop.add(idx)

        if rows_to_drop:
            save_data = save_data.drop(index=list(rows_to_drop))
            logger.info(f"[{symbol}][{data_type}] Removed {len(rows_to_drop)} rows with dict/list values")

        if save_data.empty:
            return False

        # Calculate data range
        earliest = save_data[date_col].min()
        latest = save_data[date_col].max()

        # Save to parquet
        safe_symbol = symbol.replace('.', '_')
        file_path = f"{safe_symbol}_{data_type}.parquet"
        full_path = self.cache_dir / file_path

        try:
            save_data.to_parquet(full_path, compression='snappy', index=False)
        except Exception as e:
            logger.error(f"[{symbol}][{data_type}] Failed to save parquet: {e}")
            return False

        size_bytes = full_path.stat().st_size
        now = datetime.now()
        expires_at = now + timedelta(days=self.default_ttl_days)

        # Update metadata
        await conn.execute(
            """
            INSERT INTO data_ranges
            (symbol, data_type, earliest_date, latest_date, row_count, file_path, last_updated, expires_at, size_bytes)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            ON CONFLICT (symbol, data_type) DO UPDATE SET
                earliest_date = EXCLUDED.earliest_date,
                latest_date = EXCLUDED.latest_date,
                row_count = EXCLUDED.row_count,
                file_path = EXCLUDED.file_path,
                last_updated = EXCLUDED.last_updated,
                expires_at = EXCLUDED.expires_at,
                size_bytes = EXCLUDED.size_bytes
            """,
            symbol, data_type, earliest.date(), latest.date(),
            len(save_data), file_path, now, expires_at, size_bytes
        )

        # Log the update
        await conn.execute(
            """
            INSERT INTO update_logs
            (symbol, data_type, update_type, new_range_start, new_range_end, rows_added, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            symbol, data_type, 'update', earliest.date(), latest.date(),
            len(save_data), now
        )

        logger.debug(f"[{symbol}][{data_type}] Saved cache: {earliest.date()} ~ {latest.date()}, {len(save_data)} rows")
        return True

    async def _delete_data(self, conn, symbol: str, data_type: str) -> bool:
        """Delete cached data."""
        try:
            row = await conn.fetchrow(
                "SELECT file_path FROM data_ranges WHERE symbol = $1 AND data_type = $2",
                symbol, data_type
            )

            if row:
                file_path = self.cache_dir / row["file_path"]
                if file_path.exists():
                    file_path.unlink()

                await conn.execute(
                    "DELETE FROM data_ranges WHERE symbol = $1 AND data_type = $2",
                    symbol, data_type
                )

            return True
        except Exception as e:
            logger.error(f"[{symbol}][{data_type}] Failed to delete cache: {e}")
            return False

    async def get_cache_stats(self, data_type: str = None) -> Dict[str, Any]:
        """Get cache statistics."""
        if data_type:
            row = await self.pool.fetchrow(
                """
                SELECT
                    COUNT(*) as symbol_count,
                    COALESCE(SUM(row_count), 0) as total_rows,
                    COALESCE(SUM(size_bytes), 0) as total_size
                FROM data_ranges
                WHERE data_type = $1
                """,
                data_type
            )
        else:
            row = await self.pool.fetchrow(
                """
                SELECT
                    COUNT(*) as symbol_count,
                    COALESCE(SUM(row_count), 0) as total_rows,
                    COALESCE(SUM(size_bytes), 0) as total_size
                FROM data_ranges
                """
            )

        # Stats by type
        rows_by_type = await self.pool.fetch(
            """
            SELECT data_type, COUNT(*) as count, SUM(row_count) as rows
            FROM data_ranges
            GROUP BY data_type
            """
        )

        by_type = {r["data_type"]: {"count": r["count"], "rows": r["rows"]} for r in rows_by_type}

        return {
            "symbol_count": row["symbol_count"],
            "total_rows": row["total_rows"],
            "total_size_mb": round(row["total_size"] / (1024 * 1024), 2),
            "cache_dir": str(self.cache_dir),
            "by_type": by_type,
        }

    async def list_symbols(self, data_type: str = None) -> List[Dict[str, Any]]:
        """List all cached symbols."""
        if data_type:
            rows = await self.pool.fetch(
                """
                SELECT symbol, data_type, earliest_date, latest_date, row_count, last_updated, expires_at
                FROM data_ranges
                WHERE data_type = $1
                ORDER BY last_updated DESC
                """,
                data_type
            )
        else:
            rows = await self.pool.fetch(
                """
                SELECT symbol, data_type, earliest_date, latest_date, row_count, last_updated, expires_at
                FROM data_ranges
                ORDER BY last_updated DESC
                """
            )

        return [dict(row) for row in rows]

    async def clear_expired(self) -> int:
        """Clear expired data."""
        # Get expired entries
        rows = await self.pool.fetch(
            """
            SELECT symbol, data_type, file_path FROM data_ranges
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
                    "DELETE FROM data_ranges WHERE symbol = $1 AND data_type = $2",
                    row["symbol"], row["data_type"]
                )
                count += 1

        if count > 0:
            logger.debug(f"Cleared {count} expired entries")

        return count

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None

    # ==================== Convenience Methods ====================

    async def get_stock_data(
        self,
        symbol: str,
        days: int = 120,
        fetcher=None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Get stock K-line data (convenience method)."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return await self.get_data(
            symbol, start_date, end_date, fetcher,
            data_type=DataType.STOCK_BAR,
            force_refresh=force_refresh
        )

    async def get_index_data(
        self,
        index_code: str,
        days: int = 120,
        fetcher=None,
        force_refresh: bool = False,
    ) -> pd.DataFrame:
        """Get index K-line data (convenience method)."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        return await self.get_data(
            index_code, start_date, end_date, fetcher,
            data_type=DataType.INDEX_BAR,
            force_refresh=force_refresh
        )


# Singleton instance
_instance: Optional[AsyncIncrementalDataManager] = None


def get_async_incremental_manager(config: Optional[DatabaseConfig] = None) -> AsyncIncrementalDataManager:
    """Get the async incremental manager singleton."""
    global _instance
    if _instance is None:
        _instance = AsyncIncrementalDataManager(config=config)
    return _instance
