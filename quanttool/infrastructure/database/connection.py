"""PostgreSQL connection pool manager for async operations."""

import asyncio
from contextlib import asynccontextmanager
from typing import Optional, AsyncGenerator, Any

import asyncpg
from asyncpg import Pool, Connection

from .config import DatabaseConfig, get_database_config
from ...core.logging import get_logger

logger = get_logger(__name__)


class ConnectionPool:
    """
    Async PostgreSQL connection pool manager.

    Features:
    - Lazy initialization (pool created on first use)
    - Automatic reconnection
    - Connection health checks
    - Graceful shutdown
    - Context manager support for transactions

    Usage:
        pool = ConnectionPool()

        # Direct query
        async with pool.acquire() as conn:
            result = await conn.fetch("SELECT * FROM users")

        # Transaction
        async with pool.transaction() as conn:
            await conn.execute("INSERT INTO users VALUES ($1)", "user1")
            await conn.execute("INSERT INTO logs VALUES ($1)", "created")
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the connection pool manager.

        Args:
            config: Database configuration (uses default if None)
        """
        self._config = config or get_database_config()
        self._pool: Optional[Pool] = None
        self._lock = asyncio.Lock()

    async def _create_pool(self) -> Pool:
        """Create the connection pool."""
        import json

        logger.info(
            f"Creating connection pool: {self._config.host}:{self._config.port}/{self._config.database} "
            f"(min={self._config.min_pool_size}, max={self._config.max_pool_size})"
        )

        async def init_connection(conn):
            """Initialize connection with JSON codec."""
            # Set up JSONB codec
            await conn.set_type_codec(
                'jsonb',
                encoder=lambda v: json.dumps(v) if v is not None else None,
                decoder=lambda v: json.loads(v) if v else None,
                schema='pg_catalog'
            )
            # Set up JSON codec
            await conn.set_type_codec(
                'json',
                encoder=lambda v: json.dumps(v) if v is not None else None,
                decoder=lambda v: json.loads(v) if v else None,
                schema='pg_catalog'
            )

        pool = await asyncpg.create_pool(
            host=self._config.host,
            port=self._config.port,
            database=self._config.database,
            user=self._config.user,
            password=self._config.password,
            min_size=self._config.min_pool_size,
            max_size=self._config.max_pool_size,
            timeout=self._config.connection_timeout,
            command_timeout=self._config.command_timeout,
            init=init_connection,
        )

        logger.info(f"Connection pool created successfully")
        return pool

    async def get_pool(self) -> Pool:
        """Get the connection pool (lazy initialization)."""
        if self._pool is None:
            async with self._lock:
                if self._pool is None:
                    self._pool = await self._create_pool()
        return self._pool

    @asynccontextmanager
    async def acquire(self) -> AsyncGenerator[Connection, None]:
        """
        Acquire a connection from the pool.

        Yields:
            A database connection

        Usage:
            async with pool.acquire() as conn:
                result = await conn.fetch("SELECT * FROM users")
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            yield conn

    @asynccontextmanager
    async def transaction(self) -> AsyncGenerator[Connection, None]:
        """
        Acquire a connection and start a transaction.

        Automatically commits on success, rolls back on exception.

        Yields:
            A database connection with active transaction

        Usage:
            async with pool.transaction() as conn:
                await conn.execute("INSERT INTO users VALUES ($1)", "user1")
                # Auto-commits on exit, rolls back on exception
        """
        pool = await self.get_pool()
        async with pool.acquire() as conn:
            async with conn.transaction():
                yield conn

    async def execute(self, query: str, *args: Any) -> str:
        """
        Execute a query and return the status.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Status string (e.g., "INSERT 1")
        """
        async with self.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> list:
        """
        Execute a query and return all rows.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            List of Record objects
        """
        async with self.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        """
        Execute a query and return the first row.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            Record object or None
        """
        async with self.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        """
        Execute a query and return the first value of the first row.

        Args:
            query: SQL query
            *args: Query parameters

        Returns:
            First column value
        """
        async with self.acquire() as conn:
            return await conn.fetchval(query, *args)

    async def executemany(self, query: str, args_list: list) -> None:
        """
        Execute a query multiple times with different arguments.

        Args:
            query: SQL query
            args_list: List of argument tuples
        """
        async with self.acquire() as conn:
            await conn.executemany(query, args_list)

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            async with self._lock:
                if self._pool is not None:
                    await self._pool.close()
                    self._pool = None
                    logger.info("Connection pool closed")

    async def health_check(self) -> bool:
        """Check if the database connection is healthy."""
        try:
            result = await self.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    @property
    def pool_size(self) -> int:
        """Get the current pool size."""
        return self._pool.get_size() if self._pool else 0

    @property
    def idle_connections(self) -> int:
        """Get the number of idle connections."""
        return self._pool.get_idle_size() if self._pool else 0


# Singleton instance
_pool: Optional[ConnectionPool] = None


def get_connection_pool(config: Optional[DatabaseConfig] = None) -> ConnectionPool:
    """
    Get the connection pool singleton.

    Args:
        config: Database configuration (uses default if None)

    Returns:
        ConnectionPool instance
    """
    global _pool
    if _pool is None:
        _pool = ConnectionPool(config)
    return _pool


async def close_connection_pool() -> None:
    """Close the global connection pool."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
