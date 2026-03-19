"""Schema module for PostgreSQL database initialization."""

import asyncio
from pathlib import Path
from typing import Optional

from ..connection import get_connection_pool
from ...core.logging import get_logger

logger = get_logger(__name__)

SCHEMA_DIR = Path(__file__).parent

SCHEMAS = {
    "meta": "meta_schema.sql",
    "incremental": "incremental_schema.sql",
    "cache": "cache_schema.sql",
}


async def init_database(schemas: Optional[list] = None) -> None:
    """
    Initialize the PostgreSQL database with required schemas.

    Args:
        schemas: List of schema names to initialize (default: all)
    """
    pool = get_connection_pool()
    schemas_to_init = schemas or list(SCHEMAS.keys())

    for schema_name in schemas_to_init:
        if schema_name not in SCHEMAS:
            logger.warning(f"Unknown schema: {schema_name}")
            continue

        schema_file = SCHEMA_DIR / SCHEMAS[schema_name]
        if not schema_file.exists():
            logger.error(f"Schema file not found: {schema_file}")
            continue

        logger.info(f"Initializing schema: {schema_name}")

        with open(schema_file, "r") as f:
            sql = f.read()

        async with pool.acquire() as conn:
            await conn.execute(sql)
            logger.info(f"Schema {schema_name} initialized successfully")


async def drop_all_tables() -> None:
    """Drop all tables (use with caution!)."""
    pool = get_connection_pool()

    tables = [
        "cache_entries",
        "daily_values",
        "portfolio_holdings",
        "portfolio_backtests",
        "scan_stock_results",
        "scan_records",
        "email_configs",
        "symbols",
        "tasks",
        "experiment_runs",
        "update_logs",
        "data_ranges",
    ]

    async with pool.acquire() as conn:
        for table in tables:
            try:
                await conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                logger.info(f"Dropped table: {table}")
            except Exception as e:
                logger.warning(f"Failed to drop table {table}: {e}")


def run_init(schemas: Optional[list] = None) -> None:
    """Synchronous wrapper for init_database."""
    asyncio.run(init_database(schemas))


if __name__ == "__main__":
    import sys

    schemas = sys.argv[1:] if len(sys.argv) > 1 else None
    run_init(schemas)
    print("Database initialized successfully!")
