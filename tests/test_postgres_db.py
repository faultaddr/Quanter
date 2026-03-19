"""Tests for PostgreSQL database infrastructure."""

import asyncio
import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, patch

# Skip tests if PostgreSQL is not available
pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_POSTGRES_TESTS", "true").lower() == "true",
    reason="PostgreSQL tests disabled by default. Set SKIP_POSTGRES_TESTS=false to enable."
)


class TestDatabaseConfig:
    """Test database configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        from quanttool.infrastructure.database.config import DatabaseConfig

        config = DatabaseConfig()
        assert config.host == "localhost"
        assert config.port == 5432
        assert config.database == "quanttool"
        assert config.min_pool_size == 5
        assert config.max_pool_size == 20

    def test_config_from_env(self):
        """Test configuration from environment variables."""
        from quanttool.infrastructure.database.config import DatabaseConfig

        # Set environment variables
        os.environ["DB_HOST"] = "testhost"
        os.environ["DB_PORT"] = "5433"
        os.environ["DB_NAME"] = "testdb"

        config = DatabaseConfig.from_env()
        assert config.host == "testhost"
        assert config.port == 5433
        assert config.database == "testdb"

        # Clean up
        del os.environ["DB_HOST"]
        del os.environ["DB_PORT"]
        del os.environ["DB_NAME"]

    def test_dsn_generation(self):
        """Test DSN string generation."""
        from quanttool.infrastructure.database.config import DatabaseConfig

        config = DatabaseConfig(host="localhost", port=5432, database="test", user="postgres", password="secret")
        assert "postgres" in config.dsn
        assert "localhost" in config.dsn
        assert "5432" in config.dsn


class TestConnectionPool:
    """Test connection pool functionality."""

    @pytest.fixture
    async def pool(self):
        """Create a connection pool for testing."""
        from quanttool.infrastructure.database.connection import ConnectionPool
        from quanttool.infrastructure.database.config import DatabaseConfig

        config = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "quanttool"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
        )

        pool = ConnectionPool(config)
        yield pool

        await pool.close()

    @pytest.mark.asyncio
    async def test_pool_initialization(self, pool):
        """Test that pool initializes lazily."""
        assert pool._pool is None

        # Trigger initialization
        await pool.get_pool()
        assert pool._pool is not None

    @pytest.mark.asyncio
    async def test_execute_query(self, pool):
        """Test basic query execution."""
        result = await pool.fetchval("SELECT 1")
        assert result == 1

    @pytest.mark.asyncio
    async def test_transaction(self, pool):
        """Test transaction support."""
        async with pool.transaction() as conn:
            result = await conn.fetchval("SELECT 1")
            assert result == 1

    @pytest.mark.asyncio
    async def test_health_check(self, pool):
        """Test health check."""
        healthy = await pool.health_check()
        assert healthy is True


class TestAsyncMetaDB:
    """Test AsyncMetaDB functionality."""

    @pytest.fixture
    async def meta_db(self):
        """Create an AsyncMetaDB instance for testing."""
        from quanttool.infrastructure.stores.meta_db_async import AsyncMetaDB
        from quanttool.infrastructure.database.schema import init_database

        # Initialize schema
        await init_database(["meta"])

        meta_db = AsyncMetaDB()
        yield meta_db

    @pytest.mark.asyncio
    async def test_save_and_get_experiment_run(self, meta_db):
        """Test saving and retrieving experiment runs."""
        run_data = {
            "id": "test-run-001",
            "type": "test_experiment",
            "parameters": {"param1": "value1"},
            "status": "running",
        }

        await meta_db.save_experiment_run(run_data)

        retrieved = await meta_db.get_experiment_run("test-run-001")
        assert retrieved is not None
        assert retrieved["type"] == "test_experiment"
        assert retrieved["parameters"] == {"param1": "value1"}

    @pytest.mark.asyncio
    async def test_get_experiment_runs_with_filter(self, meta_db):
        """Test filtering experiment runs."""
        # Create multiple runs
        for i in range(3):
            await meta_db.save_experiment_run({
                "id": f"test-run-{i}",
                "type": "type_a" if i < 2 else "type_b",
                "status": "completed",
            })

        # Filter by type
        runs = await meta_db.get_experiment_runs(run_type="type_a")
        assert len(runs) == 2

    @pytest.mark.asyncio
    async def test_save_and_get_task(self, meta_db):
        """Test saving and retrieving tasks."""
        task_data = {
            "id": "test-task-001",
            "type": "scan",
            "status": "pending",
            "parameters": {"symbols": ["000001.SZ"]},
        }

        await meta_db.save_task(task_data)

        retrieved = await meta_db.get_task("test-task-001")
        assert retrieved is not None
        assert retrieved["type"] == "scan"

    @pytest.mark.asyncio
    async def test_upsert_symbol(self, meta_db):
        """Test upserting symbol information."""
        symbol_data = {
            "symbol": "000001.SZ",
            "name": "平安银行",
            "industry": "银行",
            "market": "深交所",
        }

        await meta_db.upsert_symbol(symbol_data)

        retrieved = await meta_db.get_symbol_info("000001.SZ")
        assert retrieved is not None
        assert retrieved["name"] == "平安银行"

    @pytest.mark.asyncio
    async def test_save_and_get_scan_record(self, meta_db):
        """Test saving and retrieving scan records."""
        scan_data = {
            "id": "test-scan-001",
            "scan_type": "stock_scan",
            "scan_date": datetime.now().date(),
            "parameters": {"market": "csi300"},
            "results": [
                {"symbol": "000001.SZ", "total_score": 85},
                {"symbol": "000002.SZ", "total_score": 80},
            ],
        }

        scan_id = await meta_db.save_scan_record(scan_data)

        retrieved = await meta_db.get_scan_record(scan_id)
        assert retrieved is not None
        assert len(retrieved["results"]) == 2


class TestAsyncIncrementalDataManager:
    """Test AsyncIncrementalDataManager functionality."""

    @pytest.fixture
    async def manager(self, tmp_path):
        """Create an AsyncIncrementalDataManager instance for testing."""
        from quanttool.infrastructure.data_providers.incremental_data_manager_async import AsyncIncrementalDataManager
        from quanttool.infrastructure.database.schema import init_database

        # Initialize schema
        await init_database(["incremental"])

        manager = AsyncIncrementalDataManager(
            cache_dir=str(tmp_path / "cache"),
        )
        yield manager

    @pytest.mark.asyncio
    async def test_get_cache_stats_empty(self, manager):
        """Test cache stats when empty."""
        stats = await manager.get_cache_stats()
        assert stats["symbol_count"] == 0
        assert stats["total_rows"] == 0

    @pytest.mark.asyncio
    async def test_list_symbols_empty(self, manager):
        """Test listing symbols when empty."""
        symbols = await manager.list_symbols()
        assert symbols == []

    @pytest.mark.asyncio
    async def test_clear_expired(self, manager):
        """Test clearing expired entries."""
        count = await manager.clear_expired()
        assert count == 0


class TestAsyncLocalDataCache:
    """Test AsyncLocalDataCache functionality."""

    @pytest.fixture
    async def cache(self, tmp_path):
        """Create an AsyncLocalDataCache instance for testing."""
        from quanttool.infrastructure.cache.local_cache_async import AsyncLocalDataCache
        from quanttool.infrastructure.database.schema import init_database

        # Initialize schema
        await init_database(["cache"])

        cache = AsyncLocalDataCache(
            cache_dir=str(tmp_path / "cache"),
        )
        yield cache

    @pytest.mark.asyncio
    async def test_cache_miss(self, cache):
        """Test cache miss."""
        result = await cache.get("000001.SZ", "2024-01-01", "2024-12-31")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_set_and_get(self, cache):
        """Test caching and retrieving data."""
        import pandas as pd

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=10),
            "close": [10.0 + i for i in range(10)],
            "volume": [1000000] * 10,
        })

        success = await cache.set("000001.SZ", "2024-01-01", "2024-01-10", df)
        assert success is True

        retrieved = await cache.get("000001.SZ", "2024-01-01", "2024-01-10")
        assert retrieved is not None
        assert len(retrieved) == 10

    @pytest.mark.asyncio
    async def test_cache_stats(self, cache):
        """Test cache statistics."""
        import pandas as pd

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5),
            "close": [10.0] * 5,
        })

        await cache.set("000001.SZ", "2024-01-01", "2024-01-05", df)

        stats = await cache.get_stats()
        assert stats["entry_count"] == 1
        assert stats["total_rows"] == 5

    @pytest.mark.asyncio
    async def test_clear_expired(self, cache):
        """Test clearing expired entries."""
        count = await cache.clear_expired()
        assert count == 0

    @pytest.mark.asyncio
    async def test_list_entries(self, cache):
        """Test listing cache entries."""
        import pandas as pd

        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5),
            "close": [10.0] * 5,
        })

        await cache.set("000001.SZ", "2024-01-01", "2024-01-05", df)

        entries = await cache.list_entries()
        assert len(entries) == 1


class TestConcurrentAccess:
    """Test concurrent database access."""

    @pytest.fixture
    async def setup_db(self):
        """Set up database for concurrent tests."""
        from quanttool.infrastructure.database.schema import init_database
        await init_database()

    @pytest.mark.asyncio
    async def test_concurrent_writes(self, setup_db):
        """Test concurrent write operations."""
        from quanttool.infrastructure.stores.meta_db_async import get_async_meta_db

        meta_db = get_async_meta_db()

        async def create_run(i):
            await meta_db.save_experiment_run({
                "id": f"concurrent-run-{i}",
                "type": "concurrent_test",
                "status": "running",
            })

        # Run 10 concurrent writes
        tasks = [create_run(i) for i in range(10)]
        await asyncio.gather(*tasks)

        # Verify all writes succeeded
        runs = await meta_db.get_experiment_runs(run_type="concurrent_test")
        assert len(runs) == 10

    @pytest.mark.asyncio
    async def test_concurrent_reads(self, setup_db):
        """Test concurrent read operations."""
        from quanttool.infrastructure.stores.meta_db_async import get_async_meta_db

        meta_db = get_async_meta_db()

        # Create a run first
        await meta_db.save_experiment_run({
            "id": "read-test-run",
            "type": "read_test",
            "status": "completed",
        })

        async def read_run():
            return await meta_db.get_experiment_run("read-test-run")

        # Run 20 concurrent reads
        tasks = [read_run() for _ in range(20)]
        results = await asyncio.gather(*tasks)

        # All reads should succeed
        assert all(r is not None for r in results)


class TestMigration:
    """Test migration functionality."""

    @pytest.mark.asyncio
    async def test_migration_dry_run(self, tmp_path):
        """Test migration dry run."""
        from quanttool.infrastructure.database.migration import run_migration

        # Create an empty SQLite database
        import sqlite3
        db_path = tmp_path / "meta.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE experiment_runs (
                id TEXT PRIMARY KEY,
                type TEXT
            )
        """)
        conn.execute("INSERT INTO experiment_runs (id, type) VALUES ('test-001', 'test')")
        conn.commit()
        conn.close()

        # Run dry run migration
        results = await run_migration(
            sqlite_dir=str(tmp_path),
            dry_run=True,
            verify=False,
        )

        assert results["meta_db"]["status"] == "dry_run"
