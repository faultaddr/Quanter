"""Rollback tool to export PostgreSQL data back to SQLite.

This is useful for:
1. Backup before major changes
2. Testing migration in reverse
3. Fallback if PostgreSQL has issues
"""

import asyncio
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from ..connection import get_connection_pool
from ..config import DatabaseConfig
from ...core.logging import get_logger

logger = get_logger(__name__)


class PostgresToSQLiteExporter:
    """Export PostgreSQL data to SQLite."""

    def __init__(
        self,
        output_dir: str = "./backup_sqlite",
        config: Optional[DatabaseConfig] = None,
    ):
        """
        Initialize the exporter.

        Args:
            output_dir: Directory for SQLite output files
            config: PostgreSQL configuration
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._config = config
        self._pool = None

    @property
    def pool(self):
        """Get the connection pool."""
        if self._pool is None:
            self._pool = get_connection_pool(self._config)
        return self._pool

    async def export_all(self) -> Dict[str, Any]:
        """Export all data from PostgreSQL to SQLite."""
        results = {}

        # Export MetaDB tables
        results["meta_db"] = await self.export_meta_db()

        # Export IncrementalDataManager tables
        results["incremental"] = await self.export_incremental_db()

        # Export cache tables
        results["cache"] = await self.export_cache_db()

        return results

    async def export_meta_db(self) -> Dict[str, Any]:
        """Export MetaDB tables to SQLite."""
        output_path = self.output_dir / "meta.db"
        logger.info(f"Exporting MetaDB to {output_path}")

        conn = sqlite3.connect(str(output_path))
        stats = {"tables": {}, "total_rows": 0}

        # Create tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS experiment_runs (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                parameters TEXT,
                git_commit TEXT,
                data_version TEXT,
                start_time TEXT,
                end_time TEXT,
                status TEXT DEFAULT 'pending',
                results TEXT,
                artifacts TEXT
            );

            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                parameters TEXT,
                created_at TEXT,
                started_at TEXT,
                completed_at TEXT,
                result TEXT,
                error TEXT
            );

            CREATE TABLE IF NOT EXISTS symbols (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                industry TEXT,
                market TEXT,
                list_date TEXT,
                delist_date TEXT,
                status TEXT DEFAULT 'active',
                metadata TEXT
            );

            CREATE TABLE IF NOT EXISTS scan_records (
                id TEXT PRIMARY KEY,
                scan_type TEXT,
                scan_date TEXT,
                parameters TEXT,
                total_count INTEGER,
                status TEXT,
                created_at TEXT,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS scan_stock_results (
                id TEXT PRIMARY KEY,
                scan_id TEXT,
                symbol TEXT,
                score REAL,
                rank INTEGER,
                data TEXT,
                created_at TEXT,
                UNIQUE(scan_id, symbol)
            );

            CREATE TABLE IF NOT EXISTS portfolio_backtests (
                id TEXT PRIMARY KEY,
                name TEXT,
                strategy TEXT,
                parameters TEXT,
                start_date TEXT,
                end_date TEXT,
                initial_capital REAL,
                final_capital REAL,
                total_return REAL,
                annualized_return REAL,
                max_drawdown REAL,
                sharpe_ratio REAL,
                status TEXT,
                created_at TEXT,
                completed_at TEXT
            );

            CREATE TABLE IF NOT EXISTS portfolio_holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id TEXT,
                symbol TEXT,
                quantity REAL,
                entry_price REAL,
                entry_date TEXT,
                exit_price REAL,
                exit_date TEXT,
                pnl REAL
            );

            CREATE TABLE IF NOT EXISTS daily_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id TEXT,
                date TEXT,
                portfolio_value REAL,
                cash_value REAL,
                positions_value REAL,
                daily_return REAL,
                cumulative_return REAL,
                UNIQUE(backtest_id, date)
            );

            CREATE TABLE IF NOT EXISTS email_configs (
                id TEXT PRIMARY KEY,
                name TEXT,
                smtp_host TEXT,
                smtp_port INTEGER,
                smtp_user TEXT,
                smtp_password TEXT,
                from_address TEXT,
                to_addresses TEXT,
                enabled INTEGER,
                created_at TEXT,
                updated_at TEXT
            );
        """)

        # Export each table
        table_mappings = {
            "experiment_runs": "experiment_runs",
            "tasks": "tasks",
            "symbols": "symbols",
            "scan_records": "scan_records",
            "scan_stock_results": "scan_stock_results",
            "portfolio_backtests": "portfolio_backtests",
            "portfolio_holdings": "portfolio_holdings",
            "daily_values": "daily_values",
            "email_configs": "email_configs",
        }

        for pg_table, sqlite_table in table_mappings.items():
            count = await self._export_table(conn, pg_table, sqlite_table)
            stats["tables"][sqlite_table] = count
            stats["total_rows"] += count

        conn.commit()
        conn.close()

        stats["output_path"] = str(output_path)
        stats["status"] = "success"
        return stats

    async def _export_table(self, sqlite_conn, pg_table: str, sqlite_table: str) -> int:
        """Export a single table from PostgreSQL to SQLite."""
        rows = await self.pool.fetch(f"SELECT * FROM {pg_table}")

        if not rows:
            return 0

        columns = list(rows[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        col_names = ", ".join(columns)

        for row in rows:
            values = []
            for col in columns:
                val = row[col]
                # Convert JSONB to JSON string
                if isinstance(val, (dict, list)):
                    val = json.dumps(val)
                elif isinstance(val, datetime):
                    val = val.isoformat()
                values.append(val)

            sqlite_conn.execute(
                f"INSERT OR REPLACE INTO {sqlite_table} ({col_names}) VALUES ({placeholders})",
                values
            )

        logger.info(f"  Exported {len(rows)} rows from {pg_table} to {sqlite_table}")
        return len(rows)

    async def export_incremental_db(self) -> Dict[str, Any]:
        """Export IncrementalDataManager data to SQLite."""
        output_path = self.output_dir / "data_meta.db"
        logger.info(f"Exporting IncrementalDataManager to {output_path}")

        conn = sqlite3.connect(str(output_path))
        stats = {"tables": {}, "total_rows": 0}

        # Create tables
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS data_ranges (
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL DEFAULT 'stock_bar',
                earliest_date TEXT NOT NULL,
                latest_date TEXT NOT NULL,
                row_count INTEGER NOT NULL,
                file_path TEXT NOT NULL,
                last_updated TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                size_bytes INTEGER DEFAULT 0,
                PRIMARY KEY (symbol, data_type)
            );

            CREATE TABLE IF NOT EXISTS update_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                data_type TEXT NOT NULL DEFAULT 'stock_bar',
                update_type TEXT NOT NULL,
                old_range_start TEXT,
                old_range_end TEXT,
                new_range_start TEXT,
                new_range_end TEXT,
                rows_added INTEGER,
                timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_expires_at ON data_ranges(expires_at);
            CREATE INDEX IF NOT EXISTS idx_symbol ON update_log(symbol);
        """)

        # Export data_ranges
        count = await self._export_table(conn, "data_ranges", "data_ranges")
        stats["tables"]["data_ranges"] = count
        stats["total_rows"] += count

        # Export update_logs
        count = await self._export_table(conn, "update_logs", "update_log")
        stats["tables"]["update_log"] = count
        stats["total_rows"] += count

        conn.commit()
        conn.close()

        stats["output_path"] = str(output_path)
        stats["status"] = "success"
        return stats

    async def export_cache_db(self) -> Dict[str, Any]:
        """Export cache data to SQLite."""
        output_path = self.output_dir / "cache_meta.db"
        logger.info(f"Exporting LocalDataCache to {output_path}")

        conn = sqlite3.connect(str(output_path))
        stats = {"tables": {}, "total_rows": 0}

        # Create table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS cache_meta (
                cache_key TEXT PRIMARY KEY,
                file_path TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                data_hash TEXT,
                row_count INTEGER,
                size_bytes INTEGER,
                symbol TEXT,
                start_date TEXT,
                end_date TEXT,
                timeframe TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_expires_at ON cache_meta(expires_at);
            CREATE INDEX IF NOT EXISTS idx_symbol ON cache_meta(symbol);
        """)

        # Export data
        count = await self._export_table(conn, "cache_entries", "cache_meta")
        stats["tables"]["cache_meta"] = count
        stats["total_rows"] += count

        conn.commit()
        conn.close()

        stats["output_path"] = str(output_path)
        stats["status"] = "success"
        return stats

    async def close(self) -> None:
        """Close connections."""
        if self._pool:
            await self._pool.close()


async def run_rollback(
    output_dir: str = "./backup_sqlite",
) -> Dict[str, Any]:
    """
    Run the rollback export from PostgreSQL to SQLite.

    Args:
        output_dir: Directory for SQLite output files

    Returns:
        Export summary
    """
    exporter = PostgresToSQLiteExporter(output_dir)

    try:
        results = await exporter.export_all()
        return results
    finally:
        await exporter.close()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Export PostgreSQL to SQLite")
    parser.add_argument(
        "--output-dir",
        default="./backup_sqlite",
        help="Directory for SQLite output files",
    )

    args = parser.parse_args()

    results = asyncio.run(run_rollback(output_dir=args.output_dir))

    print("\nRollback Export Results:")
    print("=" * 50)
    for db_name, stats in results.items():
        print(f"\n{db_name}:")
        print(f"  Status: {stats.get('status')}")
        print(f"  Output: {stats.get('output_path')}")
        print(f"  Total rows: {stats.get('total_rows', 0)}")
        if "tables" in stats:
            for table, count in stats["tables"].items():
                print(f"    {table}: {count}")


if __name__ == "__main__":
    main()
