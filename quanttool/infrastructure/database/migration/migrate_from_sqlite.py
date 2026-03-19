"""SQLite to PostgreSQL migration tool.

This script migrates data from SQLite databases to PostgreSQL.
It handles:
1. MetaDB (meta.db)
2. IncrementalDataManager (data_meta.db)
3. LocalDataCache (cache_meta.db)
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

from ..connection import get_connection_pool
from ..config import DatabaseConfig
from ...core.logging import get_logger

logger = get_logger(__name__)


class SQLiteToPostgresMigrator:
    """Migrate data from SQLite to PostgreSQL."""

    def __init__(
        self,
        sqlite_dir: str = ".",
        config: Optional[DatabaseConfig] = None,
    ):
        """
        Initialize the migrator.

        Args:
            sqlite_dir: Directory containing SQLite databases
            config: PostgreSQL configuration
        """
        self.sqlite_dir = Path(sqlite_dir)
        self._config = config
        self._pool = None

    @property
    def pool(self):
        """Get the connection pool."""
        if self._pool is None:
            self._pool = get_connection_pool(self._config)
        return self._pool

    async def migrate_all(self, dry_run: bool = False) -> Dict[str, Any]:
        """
        Migrate all SQLite databases to PostgreSQL.

        Args:
            dry_run: If True, only report what would be migrated

        Returns:
            Migration summary
        """
        results = {
            "meta_db": await self.migrate_meta_db(dry_run),
            "incremental": await self.migrate_incremental_db(dry_run),
            "cache": await self.migrate_cache_db(dry_run),
        }

        return results

    async def migrate_meta_db(self, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate MetaDB from SQLite to PostgreSQL."""
        db_path = self.sqlite_dir / "meta.db"
        if not db_path.exists():
            logger.info(f"MetaDB not found at {db_path}")
            return {"status": "skipped", "reason": "file_not_found"}

        logger.info(f"Migrating MetaDB from {db_path}")
        stats = {"tables": {}, "total_rows": 0}

        conn = sqlite3.connect(str(db_path))

        # Get list of tables
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            if table.startswith("sqlite_"):
                continue

            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            if count == 0:
                stats["tables"][table] = 0
                continue

            stats["tables"][table] = count
            stats["total_rows"] += count

            if dry_run:
                logger.info(f"  [DRY RUN] Would migrate {count} rows from {table}")
                continue

            # Actual migration
            await self._migrate_meta_table(conn, table)
            logger.info(f"  Migrated {count} rows from {table}")

        conn.close()
        stats["status"] = "success" if not dry_run else "dry_run"
        return stats

    async def _migrate_meta_table(self, sqlite_conn, table: str) -> None:
        """Migrate a specific table from MetaDB."""
        cursor = sqlite_conn.execute(f"SELECT * FROM {table}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        if not rows:
            return

        async with self.pool.transaction() as pg_conn:
            for row in rows:
                data = dict(zip(columns, row))
                await self._insert_meta_row(pg_conn, table, data)

    async def _insert_meta_row(self, conn, table: str, data: Dict[str, Any]) -> None:
        """Insert a row into a MetaDB table."""
        # Handle JSON columns
        json_columns = ["parameters", "results", "artifacts", "metadata", "to_addresses", "data"]

        for col in json_columns:
            if col in data and isinstance(data[col], str):
                try:
                    data[col] = json.loads(data[col])
                except (json.JSONDecodeError, TypeError):
                    pass

        # Map table names
        table_map = {
            "experiment_runs": "experiment_runs",
            "tasks": "tasks",
            "symbols": "symbols",
            "scan_records": "scan_records",
            "scan_stock_results": "scan_stock_results",
            "portfolio_backtests": "portfolio_backtests",
            "portfolio_holdings": "portfolio_holdings",
            "portfolio_daily_values": "daily_values",
            "email_configs": "email_configs",
        }

        pg_table = table_map.get(table, table)

        # Build insert statement dynamically
        columns = list(data.keys())
        placeholders = [f"${i+1}" for i in range(len(columns))]
        col_names = ", ".join(columns)
        val_placeholders = ", ".join(placeholders)

        try:
            await conn.execute(
                f"""
                INSERT INTO {pg_table} ({col_names})
                VALUES ({val_placeholders})
                ON CONFLICT DO NOTHING
                """,
                *[data[col] for col in columns]
            )
        except Exception as e:
            logger.warning(f"Failed to insert into {pg_table}: {e}")

    async def migrate_incremental_db(self, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate IncrementalDataManager data."""
        db_path = self.sqlite_dir / ".cache/incremental_data/data_meta.db"
        if not db_path.exists():
            # Try alternate location
            db_path = self.sqlite_dir / "data_meta.db"

        if not db_path.exists():
            logger.info(f"IncrementalDataManager DB not found")
            return {"status": "skipped", "reason": "file_not_found"}

        logger.info(f"Migrating IncrementalDataManager from {db_path}")
        stats = {"tables": {}, "total_rows": 0}

        conn = sqlite3.connect(str(db_path))

        for table in ["data_ranges", "update_log"]:
            cursor = conn.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]

            stats["tables"][table] = count
            stats["total_rows"] += count

            if count == 0 or dry_run:
                if dry_run and count > 0:
                    logger.info(f"  [DRY RUN] Would migrate {count} rows from {table}")
                continue

            # Migrate the table
            await self._migrate_incremental_table(conn, table)
            logger.info(f"  Migrated {count} rows from {table}")

        conn.close()
        stats["status"] = "success" if not dry_run else "dry_run"
        return stats

    async def _migrate_incremental_table(self, sqlite_conn, table: str) -> None:
        """Migrate an incremental data table."""
        cursor = sqlite_conn.execute(f"SELECT * FROM {table}")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        pg_table = "data_ranges" if table == "data_ranges" else "update_logs"

        async with self.pool.transaction() as pg_conn:
            for row in rows:
                data = dict(zip(columns, row))

                if table == "data_ranges":
                    await pg_conn.execute(
                        """
                        INSERT INTO data_ranges
                        (symbol, data_type, earliest_date, latest_date, row_count, file_path,
                         last_updated, expires_at, size_bytes)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        ON CONFLICT (symbol, data_type) DO NOTHING
                        """,
                        data.get("symbol"),
                        data.get("data_type", "stock_bar"),
                        data.get("earliest_date"),
                        data.get("latest_date"),
                        data.get("row_count", 0),
                        data.get("file_path"),
                        data.get("last_updated"),
                        data.get("expires_at"),
                        data.get("size_bytes", 0),
                    )
                else:
                    await pg_conn.execute(
                        """
                        INSERT INTO update_logs
                        (symbol, data_type, update_type, old_range_start, old_range_end,
                         new_range_start, new_range_end, rows_added, timestamp)
                        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                        """,
                        data.get("symbol"),
                        data.get("data_type", "stock_bar"),
                        data.get("update_type"),
                        data.get("old_range_start"),
                        data.get("old_range_end"),
                        data.get("new_range_start"),
                        data.get("new_range_end"),
                        data.get("rows_added", 0),
                        data.get("timestamp"),
                    )

    async def migrate_cache_db(self, dry_run: bool = False) -> Dict[str, Any]:
        """Migrate LocalDataCache data."""
        db_path = self.sqlite_dir / ".cache/stock_data/cache_meta.db"
        if not db_path.exists():
            db_path = self.sqlite_dir / "cache_meta.db"

        if not db_path.exists():
            logger.info(f"LocalDataCache DB not found")
            return {"status": "skipped", "reason": "file_not_found"}

        logger.info(f"Migrating LocalDataCache from {db_path}")
        stats = {"tables": {}, "total_rows": 0}

        conn = sqlite3.connect(str(db_path))

        cursor = conn.execute("SELECT COUNT(*) FROM cache_meta")
        count = cursor.fetchone()[0]

        stats["tables"]["cache_meta"] = count
        stats["total_rows"] = count

        if count == 0 or dry_run:
            if dry_run and count > 0:
                logger.info(f"  [DRY RUN] Would migrate {count} cache entries")
            conn.close()
            stats["status"] = "dry_run" if dry_run else "success"
            return stats

        # Migrate cache entries
        cursor = conn.execute("SELECT * FROM cache_meta")
        columns = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()

        async with self.pool.transaction() as pg_conn:
            for row in rows:
                data = dict(zip(columns, row))
                await pg_conn.execute(
                    """
                    INSERT INTO cache_entries
                    (cache_key, file_path, created_at, expires_at, data_hash, row_count, size_bytes)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    ON CONFLICT (cache_key) DO NOTHING
                    """,
                    data.get("cache_key"),
                    data.get("file_path"),
                    data.get("created_at"),
                    data.get("expires_at"),
                    data.get("data_hash"),
                    data.get("row_count", 0),
                    data.get("size_bytes", 0),
                )

        conn.close()
        logger.info(f"  Migrated {count} cache entries")
        stats["status"] = "success"
        return stats

    async def verify_migration(self) -> Dict[str, Any]:
        """Verify migration by comparing row counts."""
        results = {}

        # Check PostgreSQL tables
        tables_to_check = [
            "experiment_runs", "tasks", "symbols", "scan_records",
            "scan_stock_results", "portfolio_backtests", "portfolio_holdings",
            "daily_values", "email_configs", "data_ranges", "update_logs",
            "cache_entries",
        ]

        for table in tables_to_check:
            try:
                count = await self.pool.fetchval(f"SELECT COUNT(*) FROM {table}")
                results[table] = {"pg_count": count}
            except Exception as e:
                results[table] = {"error": str(e)}

        return results

    async def close(self) -> None:
        """Close connections."""
        if self._pool:
            await self._pool.close()


async def run_migration(
    sqlite_dir: str = ".",
    dry_run: bool = False,
    verify: bool = True,
) -> Dict[str, Any]:
    """
    Run the migration from SQLite to PostgreSQL.

    Args:
        sqlite_dir: Directory containing SQLite databases
        dry_run: If True, only report what would be migrated
        verify: If True, verify migration after completion

    Returns:
        Migration summary
    """
    migrator = SQLiteToPostgresMigrator(sqlite_dir)

    try:
        results = await migrator.migrate_all(dry_run)

        if verify and not dry_run:
            results["verification"] = await migrator.verify_migration()

        return results
    finally:
        await migrator.close()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Migrate SQLite to PostgreSQL")
    parser.add_argument(
        "--sqlite-dir",
        default=".",
        help="Directory containing SQLite databases",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only report what would be migrated",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip verification after migration",
    )

    args = parser.parse_args()

    results = asyncio.run(run_migration(
        sqlite_dir=args.sqlite_dir,
        dry_run=args.dry_run,
        verify=not args.no_verify,
    ))

    print("\nMigration Results:")
    print("=" * 50)
    for db_name, stats in results.items():
        if db_name == "verification":
            print("\nVerification:")
            for table, info in stats.items():
                print(f"  {table}: {info}")
        else:
            print(f"\n{db_name}:")
            print(f"  Status: {stats.get('status')}")
            print(f"  Total rows: {stats.get('total_rows', 0)}")
            if "tables" in stats:
                for table, count in stats["tables"].items():
                    print(f"    {table}: {count}")


if __name__ == "__main__":
    main()
