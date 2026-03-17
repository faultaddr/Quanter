"""Local data cache system for QuantTool."""

import os
import json
import hashlib
import sqlite3
import pandas as pd
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

from ...core.logging import get_logger

logger = get_logger(__name__)


class LocalDataCache:
    """
    Local data cache system for stock data.

    Features:
    - SQLite metadata + Parquet data storage
    - TTL-based expiration
    - Incremental updates support
    - Thread-safe operations with locking
    """

    def __init__(
        self,
        cache_dir: str = ".cache/stock_data",
        default_ttl: int = 86400,
        max_size_mb: int = 1024
    ):
        """
        Initialize the local cache.

        Args:
            cache_dir: Directory for cache storage
            default_ttl: Default time-to-live in seconds (default: 1 day)
            max_size_mb: Maximum cache size in MB (default: 1GB)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.default_ttl = default_ttl
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self._lock = threading.Lock()

        self._init_db()
        logger.debug(f"LocalDataCache initialized at {self.cache_dir}")

    def _init_db(self) -> None:
        """Initialize SQLite metadata database."""
        self.db_path = self.cache_dir / "cache_meta.db"
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_meta (
                cache_key TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                created_at TIMESTAMP NOT NULL,
                expires_at TIMESTAMP NOT NULL,
                data_hash TEXT,
                row_count INTEGER,
                size_bytes INTEGER
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_meta(expires_at)
        """)
        self.conn.commit()

    def _generate_key(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d"
    ) -> str:
        """Generate a unique cache key."""
        key_str = f"{symbol}_{start_date}_{end_date}_{timeframe}"
        return hashlib.md5(key_str.encode()).hexdigest()

    def get(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        timeframe: str = "1d"
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

        with self._lock:
            cursor = self.conn.execute(
                "SELECT file_path, expires_at, row_count FROM cache_meta WHERE cache_key = ?",
                (key,)
            )
            row = cursor.fetchone()

            if row is None:
                logger.debug(f"Cache miss for {symbol}")
                return None

            file_path, expires_at, row_count = row
            expires_dt = datetime.fromisoformat(expires_at)

            if datetime.now() > expires_dt:
                logger.debug(f"Cache expired for {symbol}")
                self._delete_unlocked(key)
                return None

            full_path = self.cache_dir / file_path
            if not full_path.exists():
                logger.warning(f"Cache file missing for {symbol}: {file_path}")
                self._delete_unlocked(key)
                return None

        try:
            df = pd.read_parquet(full_path)
            logger.debug(f"Cache hit for {symbol}: {len(df)} rows")
            return df
        except Exception as e:
            logger.error(f"Failed to read cache for {symbol}: {e}")
            return None

    def set(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        data: pd.DataFrame,
        timeframe: str = "1d",
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache data for a symbol.

        Args:
            symbol: Stock symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            data: DataFrame to cache
            timeframe: Data timeframe
            ttl: Time-to-live in seconds (uses default if None)

        Returns:
            True if cached successfully, False otherwise
        """
        if data is None or data.empty:
            logger.warning(f"Attempted to cache empty data for {symbol}")
            return False

        key = self._generate_key(symbol, start_date, end_date, timeframe)
        ttl = ttl or self.default_ttl

        # Generate file path
        file_path = f"{key}.parquet"
        full_path = self.cache_dir / file_path

        try:
            # 清理无法序列化的列（如 dict, list 类型）
            data_to_cache = data.copy()
            cols_to_drop = []

            for col in data_to_cache.columns:
                # 检查整列中是否有不可序列化的类型
                try:
                    # 尝试对列进行 hash 测试
                    sample = data_to_cache[col].dropna()
                    if len(sample) > 0:
                        # 检查所有非空值
                        for val in sample.values:
                            if isinstance(val, (dict, list, set)):
                                cols_to_drop.append(col)
                                logger.debug(f"Dropping unhashable column '{col}' (contains {type(val).__name__}) from cache data for {symbol}")
                                break
                except Exception:
                    # 如果检查失败，安全起见也删除
                    cols_to_drop.append(col)
                    logger.debug(f"Dropping column '{col}' (type check failed) from cache data for {symbol}")

            if cols_to_drop:
                data_to_cache = data_to_cache.drop(columns=cols_to_drop)

            # 检查是否还有列剩下
            if data_to_cache.empty or len(data_to_cache.columns) == 0:
                logger.warning(f"No hashable columns left for {symbol}, skipping cache")
                return False

            # Save data as Parquet (no lock needed for file write)
            data_to_cache.to_parquet(full_path, compression='snappy', index=False)

            # Get file size
            size_bytes = full_path.stat().st_size

            # Calculate data hash with error handling
            try:
                data_hash = hashlib.md5(
                    pd.util.hash_pandas_object(data_to_cache).values.tobytes()
                ).hexdigest()
            except Exception as hash_err:
                logger.debug(f"Could not compute hash for {symbol}: {hash_err}")
                data_hash = "no_hash"

            # Update metadata with lock
            now = datetime.now()
            expires_at = now + timedelta(seconds=ttl)

            with self._lock:
                self.conn.execute("""
                    INSERT OR REPLACE INTO cache_meta
                    (cache_key, file_path, created_at, expires_at, data_hash, row_count, size_bytes)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    key, file_path, now.isoformat(), expires_at.isoformat(),
                    data_hash, len(data_to_cache), size_bytes
                ))
                self.conn.commit()

            logger.debug(f"Cached {len(data_to_cache)} rows for {symbol} ({size_bytes} bytes)")
            return True

        except Exception as e:
            logger.error(f"Failed to cache data for {symbol}: {e}")
            return False

    def delete(self, key: str) -> bool:
        """Delete a cache entry by key."""
        with self._lock:
            return self._delete_unlocked(key)

    def _delete_unlocked(self, key: str) -> bool:
        """Delete a cache entry by key (assumes lock is held)."""
        try:
            cursor = self.conn.execute(
                "SELECT file_path FROM cache_meta WHERE cache_key = ?",
                (key,)
            )
            row = cursor.fetchone()

            if row:
                file_path = row[0]
                full_path = self.cache_dir / file_path
                if full_path.exists():
                    full_path.unlink()

                self.conn.execute("DELETE FROM cache_meta WHERE cache_key = ?", (key,))
                self.conn.commit()

            return True
        except Exception as e:
            logger.error(f"Failed to delete cache entry {key}: {e}")
            return False

    def clear_expired(self) -> int:
        """Clear all expired cache entries."""
        with self._lock:
            now = datetime.now().isoformat()
            cursor = self.conn.execute(
                "SELECT cache_key, file_path FROM cache_meta WHERE expires_at < ?",
                (now,)
            )
            expired = cursor.fetchall()

            count = 0
            for key, file_path in expired:
                full_path = self.cache_dir / file_path
                if full_path.exists():
                    full_path.unlink()
                self.conn.execute("DELETE FROM cache_meta WHERE cache_key = ?", (key,))
                count += 1

            self.conn.commit()
            if count > 0:
                logger.debug(f"Cleared {count} expired cache entries")

            return count

    def clear_all(self) -> int:
        """Clear all cache entries."""
        with self._lock:
            cursor = self.conn.execute("SELECT file_path FROM cache_meta")
            files = cursor.fetchall()

            count = 0
            for (file_path,) in files:
                full_path = self.cache_dir / file_path
                if full_path.exists():
                    full_path.unlink()
                    count += 1

            self.conn.execute("DELETE FROM cache_meta")
            self.conn.commit()

            logger.debug(f"Cleared all cache: {count} files")
            return count

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            cursor = self.conn.execute("""
                SELECT
                    COUNT(*) as entry_count,
                    COALESCE(SUM(row_count), 0) as total_rows,
                    COALESCE(SUM(size_bytes), 0) as total_size_bytes
                FROM cache_meta
            """)
            row = cursor.fetchone()

            return {
                "entry_count": row[0],
                "total_rows": row[1],
                "total_size_bytes": row[2],
                "total_size_mb": round(row[2] / (1024 * 1024), 2),
                "cache_dir": str(self.cache_dir)
            }

    def list_entries(self, limit: int = 100) -> List[Dict[str, Any]]:
        """List cache entries."""
        with self._lock:
            cursor = self.conn.execute("""
                SELECT cache_key, file_path, created_at, expires_at, row_count, size_bytes
                FROM cache_meta
                ORDER BY created_at DESC
                LIMIT ?
            """, (limit,))

            entries = []
            for row in cursor.fetchall():
                entries.append({
                    "key": row[0],
                    "file": row[1],
                    "created": row[2],
                    "expires": row[3],
                    "rows": row[4],
                    "size_bytes": row[5]
                })

            return entries

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            logger.debug("LocalDataCache connection closed")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()