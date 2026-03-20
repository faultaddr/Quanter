"""Async PostgreSQL-based metadata database for QuantTool.

This module provides an async interface for metadata storage using PostgreSQL.
It replaces the SQLite-based MetaDB with better concurrency support.
"""

import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional

from ..database.connection import get_connection_pool
from ..database.config import DatabaseConfig
from ...core.logging import get_logger

logger = get_logger(__name__)


class AsyncMetaDB:
    """
    Async PostgreSQL-based metadata database for QuantTool.

    Features:
    - Full async/await support for FastAPI
    - Connection pooling for high concurrency
    - JSONB for efficient JSON operations
    - Transaction support for data integrity
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        """
        Initialize the async metadata database.

        Args:
            config: Database configuration (uses default if None)
        """
        self._config = config
        self._pool = None

    @property
    def pool(self):
        """Get the connection pool (lazy initialization)."""
        if self._pool is None:
            self._pool = get_connection_pool(self._config)
        return self._pool

    # ==================== Experiment Run Methods ====================

    async def save_experiment_run(self, run_data: Dict[str, Any]) -> None:
        """Save experiment run data to the database."""
        async with self.pool.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO experiment_runs
                (id, type, parameters, git_commit, data_version, start_time, end_time, status, results, artifacts)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (id) DO UPDATE SET
                    type = EXCLUDED.type,
                    parameters = EXCLUDED.parameters,
                    git_commit = EXCLUDED.git_commit,
                    data_version = EXCLUDED.data_version,
                    start_time = EXCLUDED.start_time,
                    end_time = EXCLUDED.end_time,
                    status = EXCLUDED.status,
                    results = EXCLUDED.results,
                    artifacts = EXCLUDED.artifacts,
                    updated_at = NOW()
                """,
                run_data.get("id") or str(uuid.uuid4()),
                run_data.get("type"),
                run_data.get("parameters", {}),
                run_data.get("git_commit"),
                run_data.get("data_version"),
                run_data.get("start_time"),
                run_data.get("end_time"),
                run_data.get("status", "pending"),
                run_data.get("results"),
                run_data.get("artifacts"),
            )
        logger.info(f"Saved experiment run: {run_data.get('id')}")

    async def get_experiment_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment run data by ID."""
        row = await self.pool.fetchrow(
            """
            SELECT id, type, parameters, git_commit, data_version,
                   start_time, end_time, status, results, artifacts
            FROM experiment_runs WHERE id = $1
            """,
            run_id,
        )

        if row:
            return dict(row)
        return None

    async def get_experiment_runs(
        self, run_type: str = None, status: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve experiment runs with optional filters."""
        conditions = []
        params = []
        param_idx = 1

        if run_type:
            conditions.append(f"type = ${param_idx}")
            params.append(run_type)
            param_idx += 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        rows = await self.pool.fetch(
            f"""
            SELECT id, type, parameters, git_commit, data_version,
                   start_time, end_time, status, results, artifacts
            FROM experiment_runs WHERE {where_clause}
            ORDER BY start_time DESC LIMIT ${param_idx}
            """,
            *params,
        )

        return [dict(row) for row in rows]

    # ==================== Task Methods ====================

    async def save_task(self, task_data: Dict[str, Any]) -> None:
        """Save task data to the database."""
        async with self.pool.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO tasks
                (id, type, status, parameters, created_at, started_at, completed_at, result, error)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    type = EXCLUDED.type,
                    status = EXCLUDED.status,
                    parameters = EXCLUDED.parameters,
                    started_at = EXCLUDED.started_at,
                    completed_at = EXCLUDED.completed_at,
                    result = EXCLUDED.result,
                    error = EXCLUDED.error
                """,
                task_data.get("id") or str(uuid.uuid4()),
                task_data.get("type"),
                task_data.get("status", "pending"),
                task_data.get("parameters", {}),
                task_data.get("created_at"),
                task_data.get("started_at"),
                task_data.get("completed_at"),
                task_data.get("result"),
                task_data.get("error"),
            )
        logger.info(f"Saved task: {task_data.get('id')}")

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task data by ID."""
        row = await self.pool.fetchrow(
            """
            SELECT id, type, status, parameters, created_at, started_at, completed_at, result, error
            FROM tasks WHERE id = $1
            """,
            task_id,
        )

        if row:
            return dict(row)
        return None

    async def get_tasks(
        self, task_type: str = None, status: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve tasks with optional filters."""
        conditions = []
        params = []
        param_idx = 1

        if task_type:
            conditions.append(f"type = ${param_idx}")
            params.append(task_type)
            param_idx += 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        rows = await self.pool.fetch(
            f"""
            SELECT id, type, status, parameters, created_at, started_at, completed_at, result, error
            FROM tasks WHERE {where_clause}
            ORDER BY created_at DESC LIMIT ${param_idx}
            """,
            *params,
        )

        return [dict(row) for row in rows]

    # ==================== Symbol Methods ====================

    async def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve symbol information by symbol code."""
        row = await self.pool.fetchrow(
            """
            SELECT symbol, name, industry, market, list_date, delist_date, status, metadata
            FROM symbols WHERE symbol = $1
            """,
            symbol,
        )

        if row:
            return dict(row)
        return None

    async def get_symbols_by_filter(
        self, industry: str = None, market: str = None, status: str = "active", limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Retrieve symbols with optional filters."""
        conditions = []
        params = []
        param_idx = 1

        if industry:
            conditions.append(f"industry = ${param_idx}")
            params.append(industry)
            param_idx += 1

        if market:
            conditions.append(f"market = ${param_idx}")
            params.append(market)
            param_idx += 1

        if status:
            conditions.append(f"status = ${param_idx}")
            params.append(status)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        rows = await self.pool.fetch(
            f"""
            SELECT symbol, name, industry, market, list_date, delist_date, status, metadata
            FROM symbols WHERE {where_clause}
            ORDER BY symbol LIMIT ${param_idx}
            """,
            *params,
        )

        return [dict(row) for row in rows]

    async def upsert_symbol(self, symbol_data: Dict[str, Any]) -> None:
        """Insert or update symbol information."""
        async with self.pool.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO symbols (symbol, name, industry, market, list_date, delist_date, status, metadata)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (symbol) DO UPDATE SET
                    name = EXCLUDED.name,
                    industry = EXCLUDED.industry,
                    market = EXCLUDED.market,
                    list_date = EXCLUDED.list_date,
                    delist_date = EXCLUDED.delist_date,
                    status = EXCLUDED.status,
                    metadata = EXCLUDED.metadata,
                    updated_at = NOW()
                """,
                (
                    symbol_data.get("symbol"),
                    symbol_data.get("name"),
                    symbol_data.get("industry"),
                    symbol_data.get("market"),
                    symbol_data.get("list_date"),
                    symbol_data.get("delist_date"),
                    symbol_data.get("status", "active"),
                    symbol_data.get("metadata", {}),
                ),
            )

    # ==================== Scan Record Methods ====================

    async def save_scan_record(self, scan_data: Dict[str, Any]) -> str:
        """Save a scan record to the database."""
        scan_id = scan_data.get("id") or str(uuid.uuid4())
        scan_date = scan_data.get("scan_date", datetime.now().date())

        async with self.pool.transaction() as conn:
            # Save scan record
            await conn.execute(
                """
                INSERT INTO scan_records
                (id, scan_type, scan_date, parameters, total_count, status)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO UPDATE SET
                    scan_type = EXCLUDED.scan_type,
                    scan_date = EXCLUDED.scan_date,
                    parameters = EXCLUDED.parameters,
                    total_count = EXCLUDED.total_count,
                    status = EXCLUDED.status
                """,
                (
                    scan_id,
                    scan_data.get("scan_type", "stock_scan"),
                    scan_date,
                    scan_data.get("parameters", {}),
                    scan_data.get("total_count", 0),
                    scan_data.get("status", "completed"),
                ),
            )

            # Save individual stock results
            results = scan_data.get("results", [])
            for rank, result in enumerate(results, 1):
                await conn.execute(
                    """
                    INSERT INTO scan_stock_results
                    (scan_id, symbol, score, rank, data)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (scan_id, symbol) DO UPDATE SET
                        score = EXCLUDED.score,
                        rank = EXCLUDED.rank,
                        data = EXCLUDED.data
                    """,
                    (
                        scan_id,
                        result.get("symbol"),
                        result.get("total_score", 0),
                        rank,
                        result,
                    ),
                )

        logger.info(f"Saved scan record: {scan_id} with {len(results)} stocks")
        return scan_id

    async def get_scan_record(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a scan record by ID."""
        row = await self.pool.fetchrow(
            """
            SELECT id, scan_type, scan_date, parameters, total_count, status, created_at, completed_at
            FROM scan_records WHERE id = $1
            """,
            scan_id,
        )

        if not row:
            return None

        scan_record = dict(row)

        # Get stock results
        results = await self.pool.fetch(
            """
            SELECT symbol, score, rank, data
            FROM scan_stock_results WHERE scan_id = $1
            ORDER BY rank
            """,
            scan_id,
        )

        scan_record["results"] = [dict(r) for r in results]
        return scan_record

    async def get_scan_history(
        self,
        scan_type: str = None,
        start_date: str = None,
        end_date: str = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Retrieve scan history with optional filters."""
        conditions = []
        params = []
        param_idx = 1

        if scan_type:
            conditions.append(f"scan_type = ${param_idx}")
            params.append(scan_type)
            param_idx += 1

        if start_date:
            conditions.append(f"scan_date >= ${param_idx}")
            params.append(start_date)
            param_idx += 1

        if end_date:
            conditions.append(f"scan_date <= ${param_idx}")
            params.append(end_date)
            param_idx += 1

        where_clause = " AND ".join(conditions) if conditions else "TRUE"
        params.append(limit)

        rows = await self.pool.fetch(
            f"""
            SELECT id, scan_type, scan_date, parameters, total_count, status, created_at, completed_at
            FROM scan_records WHERE {where_clause}
            ORDER BY scan_date DESC LIMIT ${param_idx}
            """,
            *params,
        )

        return [dict(row) for row in rows]

    async def compare_scans(self, scan_id_1: str, scan_id_2: str) -> Dict[str, Any]:
        """Compare two scans and return common stocks and differences."""
        # Get stocks from both scans
        rows_1 = await self.pool.fetch(
            "SELECT symbol, score, rank FROM scan_stock_results WHERE scan_id = $1 ORDER BY rank",
            scan_id_1,
        )
        rows_2 = await self.pool.fetch(
            "SELECT symbol, score, rank FROM scan_stock_results WHERE scan_id = $1 ORDER BY rank",
            scan_id_2,
        )

        stocks_1 = {r["symbol"]: {"score": r["score"], "rank": r["rank"]} for r in rows_1}
        stocks_2 = {r["symbol"]: {"score": r["score"], "rank": r["rank"]} for r in rows_2}

        # Find common and unique stocks
        common_symbols = set(stocks_1.keys()) & set(stocks_2.keys())
        only_in_1 = set(stocks_1.keys()) - set(stocks_2.keys())
        only_in_2 = set(stocks_2.keys()) - set(stocks_1.keys())

        # Analyze common stocks
        common_stocks = [
            {
                "symbol": symbol,
                "scan_1_rank": stocks_1[symbol]["rank"],
                "scan_2_rank": stocks_2[symbol]["rank"],
                "scan_1_score": stocks_1[symbol]["score"],
                "scan_2_score": stocks_2[symbol]["score"],
                "score_change": stocks_2[symbol]["score"] - stocks_1[symbol]["score"],
                "rank_change": stocks_1[symbol]["rank"] - stocks_2[symbol]["rank"],
            }
            for symbol in common_symbols
        ]
        common_stocks.sort(key=lambda x: x["rank_change"], reverse=True)

        return {
            "scan_id_1": scan_id_1,
            "scan_id_2": scan_id_2,
            "common_stocks": common_stocks,
            "common_count": len(common_symbols),
            "only_in_scan_1": list(only_in_1),
            "only_in_scan_2": list(only_in_2),
            "only_in_scan_1_count": len(only_in_1),
            "only_in_scan_2_count": len(only_in_2),
        }

    # ==================== Portfolio Backtest Methods ====================

    async def create_portfolio_backtest(self, backtest_data: Dict[str, Any]) -> str:
        """Create a new portfolio backtest record."""
        backtest_id = backtest_data.get("id") or str(uuid.uuid4())

        async with self.pool.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO portfolio_backtests
                (id, name, strategy, parameters, start_date, end_date, initial_capital, status)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                (
                    backtest_id,
                    backtest_data.get("name", f"Portfolio_{backtest_id[:8]}"),
                    backtest_data.get("strategy", "default"),
                    backtest_data.get("parameters", {}),
                    backtest_data.get("start_date"),
                    backtest_data.get("end_date"),
                    backtest_data.get("initial_capital", 500000),
                    backtest_data.get("status", "pending"),
                ),
            )

        logger.info(f"Created portfolio backtest: {backtest_id}")
        return backtest_id

    async def add_portfolio_holding(self, holding_data: Dict[str, Any]) -> int:
        """Add a holding to a portfolio backtest."""
        row = await self.pool.fetchrow(
            """
            INSERT INTO portfolio_holdings
            (backtest_id, symbol, quantity, entry_price, entry_date)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
            """,
            (
                holding_data.get("backtest_id"),
                holding_data.get("symbol"),
                holding_data.get("quantity"),
                holding_data.get("entry_price"),
                holding_data.get("entry_date"),
            ),
        )
        return row["id"]

    async def update_holding_exit(self, holding_id: int, exit_data: Dict[str, Any]) -> None:
        """Update holding with exit information."""
        await self.pool.execute(
            """
            UPDATE portfolio_holdings
            SET exit_price = $1, exit_date = $2, pnl = $3
            WHERE id = $4
            """,
            (
                exit_data.get("exit_price"),
                exit_data.get("exit_date"),
                exit_data.get("pnl"),
                holding_id,
            ),
        )

    async def record_daily_value(self, value_data: Dict[str, Any]) -> None:
        """Record daily portfolio value."""
        async with self.pool.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO daily_values
                (backtest_id, date, portfolio_value, cash_value, positions_value, daily_return, cumulative_return)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (backtest_id, date) DO UPDATE SET
                    portfolio_value = EXCLUDED.portfolio_value,
                    cash_value = EXCLUDED.cash_value,
                    positions_value = EXCLUDED.positions_value,
                    daily_return = EXCLUDED.daily_return,
                    cumulative_return = EXCLUDED.cumulative_return
                """,
                (
                    value_data.get("backtest_id"),
                    value_data.get("date"),
                    value_data.get("portfolio_value"),
                    value_data.get("cash_value"),
                    value_data.get("positions_value"),
                    value_data.get("daily_return"),
                    value_data.get("cumulative_return"),
                ),
            )

    async def get_portfolio_backtest(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio backtest by ID with holdings."""
        row = await self.pool.fetchrow(
            """
            SELECT id, name, strategy, parameters, start_date, end_date, initial_capital, final_capital,
                   total_return, annualized_return, max_drawdown, sharpe_ratio, status, created_at, completed_at
            FROM portfolio_backtests WHERE id = $1
            """,
            backtest_id,
        )

        if not row:
            return None

        backtest = dict(row)

        # Get holdings
        holdings = await self.pool.fetch(
            """
            SELECT id, symbol, quantity, entry_price, entry_date, exit_price, exit_date, pnl
            FROM portfolio_holdings WHERE backtest_id = $1
            """,
            backtest_id,
        )
        backtest["holdings"] = [dict(h) for h in holdings]

        # Get daily values
        daily_values = await self.pool.fetch(
            """
            SELECT date, portfolio_value, cash_value, positions_value, daily_return, cumulative_return
            FROM daily_values WHERE backtest_id = $1 ORDER BY date
            """,
            backtest_id,
        )
        backtest["daily_values"] = [dict(d) for d in daily_values]

        return backtest

    async def get_active_portfolios(self) -> List[Dict[str, Any]]:
        """Get all active (non-closed) portfolio backtests."""
        rows = await self.pool.fetch(
            """
            SELECT id, name, strategy, start_date, initial_capital, status
            FROM portfolio_backtests WHERE status = 'pending' OR status = 'running'
            ORDER BY created_at DESC
            """
        )
        return [dict(row) for row in rows]

    async def close_portfolio_backtest(self, backtest_id: str, metrics: Dict[str, Any]) -> None:
        """Close a portfolio backtest and update metrics."""
        await self.pool.execute(
            """
            UPDATE portfolio_backtests
            SET status = 'completed',
                end_date = $1,
                final_capital = $2,
                total_return = $3,
                annualized_return = $4,
                max_drawdown = $5,
                sharpe_ratio = $6,
                completed_at = NOW()
            WHERE id = $7
            """,
            (
                metrics.get("end_date"),
                metrics.get("final_capital"),
                metrics.get("total_return"),
                metrics.get("annualized_return"),
                metrics.get("max_drawdown"),
                metrics.get("sharpe_ratio"),
                backtest_id,
            ),
        )

    # ==================== Email Config Methods ====================

    async def save_email_config(self, config_data: Dict[str, Any]) -> str:
        """Save email configuration."""
        config_id = config_data.get("id") or str(uuid.uuid4())

        async with self.pool.transaction() as conn:
            await conn.execute(
                """
                INSERT INTO email_configs
                (id, name, smtp_host, smtp_port, smtp_user, smtp_password, from_address, to_addresses, enabled)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (id) DO UPDATE SET
                    name = EXCLUDED.name,
                    smtp_host = EXCLUDED.smtp_host,
                    smtp_port = EXCLUDED.smtp_port,
                    smtp_user = EXCLUDED.smtp_user,
                    smtp_password = EXCLUDED.smtp_password,
                    from_address = EXCLUDED.from_address,
                    to_addresses = EXCLUDED.to_addresses,
                    enabled = EXCLUDED.enabled,
                    updated_at = NOW()
                """,
                (
                    config_id,
                    config_data.get("name", "default"),
                    config_data.get("smtp_host"),
                    config_data.get("smtp_port"),
                    config_data.get("smtp_user"),
                    config_data.get("smtp_password"),
                    config_data.get("from_address"),
                    config_data.get("to_addresses", []),
                    config_data.get("enabled", True),
                ),
            )

        return config_id

    async def get_email_config(self, config_id: str = None, name: str = None) -> Optional[Dict[str, Any]]:
        """Get email configuration."""
        if config_id:
            row = await self.pool.fetchrow(
                """
                SELECT id, name, smtp_host, smtp_port, smtp_user, smtp_password, from_address, to_addresses, enabled
                FROM email_configs WHERE id = $1
                """,
                config_id,
            )
        elif name:
            row = await self.pool.fetchrow(
                """
                SELECT id, name, smtp_host, smtp_port, smtp_user, smtp_password, from_address, to_addresses, enabled
                FROM email_configs WHERE name = $1
                """,
                name,
            )
        else:
            row = await self.pool.fetchrow(
                """
                SELECT id, name, smtp_host, smtp_port, smtp_user, smtp_password, from_address, to_addresses, enabled
                FROM email_configs WHERE enabled = true LIMIT 1
                """
            )

        if row:
            return dict(row)
        return None

    # ==================== Utility Methods ====================

    async def health_check(self) -> bool:
        """Check if database connection is healthy."""
        try:
            result = await self.pool.fetchval("SELECT 1")
            return result == 1
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False

    async def close(self) -> None:
        """Close the connection pool."""
        if self._pool:
            await self._pool.close()
            self._pool = None


# Singleton instance
_instance: Optional[AsyncMetaDB] = None


def get_async_meta_db(config: Optional[DatabaseConfig] = None) -> AsyncMetaDB:
    """Get the async MetaDB singleton."""
    global _instance
    if _instance is None:
        _instance = AsyncMetaDB(config)
    return _instance
