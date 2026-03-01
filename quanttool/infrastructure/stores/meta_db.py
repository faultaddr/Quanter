"""Simple metadata database using SQLite for QuantTool."""

import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path
from ...core.logging import get_logger


logger = get_logger(__name__)


class MetaDB:
    """SQLite-based metadata database for QuantTool."""

    def __init__(self, db_path: str = "./meta.db"):
        """
        Initialize the metadata database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = Path(db_path)
        self.init_tables()

    def init_tables(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table for experiment runs
        cursor.execute(
            """
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
            )
        """
        )

        # Table for tasks
        cursor.execute(
            """
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
            )
        """
        )

        # Table for symbols/metadata
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS symbols (
                symbol TEXT PRIMARY KEY,
                name TEXT,
                exchange TEXT,
                currency TEXT,
                sector TEXT,
                industry TEXT,
                created_at TEXT,
                updated_at TEXT
            )
        """
        )

        # Table for scan records (扫描记录)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_records (
                id TEXT PRIMARY KEY,
                scan_date TEXT NOT NULL,
                market TEXT NOT NULL,
                days_analyzed INTEGER,
                total_stocks INTEGER,
                bias_filter_min REAL,
                bias_filter_max REAL,
                created_at TEXT
            )
        """
        )

        # Table for scan stock results (扫描个股结果)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_stock_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT,
                close_price REAL,
                daily_return REAL,
                total_score INTEGER,
                rating TEXT,
                action TEXT,
                risk_level TEXT,
                bias_6 REAL,
                bias_12 REAL,
                bias_24 REAL,
                trend_score INTEGER,
                momentum_score INTEGER,
                volatility_score INTEGER,
                capital_score INTEGER,
                structure_score INTEGER,
                rank INTEGER,
                created_at TEXT,
                FOREIGN KEY (scan_id) REFERENCES scan_records(id)
            )
        """
        )

        # Table for tracking stock performance after scan (回测跟踪)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_performance_tracking (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scan_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                scan_date TEXT NOT NULL,
                scan_price REAL,
                day_1_return REAL,
                day_3_return REAL,
                day_5_return REAL,
                day_10_return REAL,
                day_20_return REAL,
                day_60_return REAL,
                max_return_20d REAL,
                max_drawdown_20d REAL,
                updated_at TEXT,
                FOREIGN KEY (scan_id) REFERENCES scan_records(id)
            )
        """
        )

        # Create indexes for better query performance
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_results_scan_id ON scan_stock_results(scan_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_results_symbol ON scan_stock_results(symbol)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_tracking_scan_id ON scan_performance_tracking(scan_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_tracking_symbol ON scan_performance_tracking(symbol)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_scan_records_date ON scan_records(scan_date)"
        )

        # Table for portfolio backtests (投资组合回测)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_backtests (
                id TEXT PRIMARY KEY,
                scan_id TEXT NOT NULL,
                portfolio_name TEXT NOT NULL,
                initial_capital REAL NOT NULL,
                current_value REAL,
                start_date TEXT NOT NULL,
                end_date TEXT,
                hold_days INTEGER DEFAULT 20,
                status TEXT DEFAULT 'active',
                total_return REAL,
                annualized_return REAL,
                sharpe_ratio REAL,
                max_drawdown REAL,
                volatility REAL,
                win_rate REAL,
                created_at TEXT,
                completed_at TEXT,
                FOREIGN KEY (scan_id) REFERENCES scan_records(id)
            )
        """
        )

        # Table for portfolio holdings (投资组合持仓明细)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_holdings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id TEXT NOT NULL,
                symbol TEXT NOT NULL,
                name TEXT,
                entry_date TEXT NOT NULL,
                entry_price REAL NOT NULL,
                shares INTEGER NOT NULL,
                weight REAL NOT NULL,
                initial_value REAL NOT NULL,
                exit_date TEXT,
                exit_price REAL,
                realized_return REAL,
                max_return REAL,
                max_drawdown REAL,
                status TEXT DEFAULT 'holding',
                created_at TEXT,
                updated_at TEXT,
                FOREIGN KEY (backtest_id) REFERENCES portfolio_backtests(id)
            )
        """
        )

        # Table for portfolio daily values (组合每日净值)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS portfolio_daily_values (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                backtest_id TEXT NOT NULL,
                date TEXT NOT NULL,
                total_value REAL NOT NULL,
                cash_value REAL NOT NULL,
                market_value REAL NOT NULL,
                daily_return REAL,
                cumulative_return REAL,
                benchmark_return REAL,
                created_at TEXT,
                FOREIGN KEY (backtest_id) REFERENCES portfolio_backtests(id),
                UNIQUE(backtest_id, date)
            )
        """
        )

        # Table for email configurations (邮件配置)
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS email_configs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                smtp_host TEXT NOT NULL,
                smtp_port INTEGER DEFAULT 587,
                username TEXT NOT NULL,
                password TEXT NOT NULL,
                from_addr TEXT NOT NULL,
                to_addrs TEXT NOT NULL,
                enabled BOOLEAN DEFAULT 1,
                created_at TEXT,
                updated_at TEXT
            )
        """
        )

        # Create indexes for new tables
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_backtests_scan_id ON portfolio_backtests(scan_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_backtests_status ON portfolio_backtests(status)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_backtest_id ON portfolio_holdings(backtest_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_holdings_symbol ON portfolio_holdings(symbol)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_daily_values_backtest_id ON portfolio_daily_values(backtest_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_portfolio_daily_values_date ON portfolio_daily_values(date)"
        )

        conn.commit()
        conn.close()

        logger.info(f"MetaDB initialized at {self.db_path}")

    def save_experiment_run(self, run_data: Dict[str, Any]):
        """
        Save experiment run data to the database.

        Args:
            run_data: Dictionary containing experiment run information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO experiment_runs
            (id, type, parameters, git_commit, data_version, start_time, end_time, status, results, artifacts)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                run_data.get("id"),
                run_data.get("type"),
                (
                    json.dumps(run_data.get("parameters"))
                    if run_data.get("parameters")
                    else None
                ),
                run_data.get("git_commit"),
                run_data.get("data_version"),
                (
                    run_data.get("start_time").isoformat()
                    if run_data.get("start_time")
                    else None
                ),
                (
                    run_data.get("end_time").isoformat()
                    if run_data.get("end_time")
                    else None
                ),
                run_data.get("status"),
                (
                    json.dumps(run_data.get("results"))
                    if run_data.get("results")
                    else None
                ),
                (
                    json.dumps(run_data.get("artifacts"))
                    if run_data.get("artifacts")
                    else None
                ),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Saved experiment run: {run_data.get('id')}")

    def get_experiment_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve experiment run data by ID.

        Args:
            run_id: ID of the experiment run

        Returns:
            Dictionary containing experiment run information or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, type, parameters, git_commit, data_version, start_time, end_time, status, results, artifacts
            FROM experiment_runs WHERE id = ?
        """,
            (run_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "type": row[1],
                "parameters": json.loads(row[2]) if row[2] else {},
                "git_commit": row[3],
                "data_version": row[4],
                "start_time": datetime.fromisoformat(row[5]) if row[5] else None,
                "end_time": datetime.fromisoformat(row[6]) if row[6] else None,
                "status": row[7],
                "results": json.loads(row[8]) if row[8] else {},
                "artifacts": json.loads(row[9]) if row[9] else [],
            }

        return None

    def get_experiment_runs(
        self, run_type: str = None, status: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve experiment runs with optional filters.

        Args:
            run_type: Type of experiment to filter by
            status: Status to filter by

        Returns:
            List of experiment run dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT id, type, parameters, git_commit, data_version, start_time, end_time, status, results, artifacts FROM experiment_runs WHERE 1=1"
        params = []

        if run_type:
            query += " AND type = ?"
            params.append(run_type)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY start_time DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        runs = []
        for row in rows:
            runs.append(
                {
                    "id": row[0],
                    "type": row[1],
                    "parameters": json.loads(row[2]) if row[2] else {},
                    "git_commit": row[3],
                    "data_version": row[4],
                    "start_time": datetime.fromisoformat(row[5]) if row[5] else None,
                    "end_time": datetime.fromisoformat(row[6]) if row[6] else None,
                    "status": row[7],
                    "results": json.loads(row[8]) if row[8] else {},
                    "artifacts": json.loads(row[9]) if row[9] else [],
                }
            )

        return runs

    def save_task(self, task_data: Dict[str, Any]):
        """
        Save task data to the database.

        Args:
            task_data: Dictionary containing task information
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO tasks
            (id, type, status, parameters, created_at, started_at, completed_at, result, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                task_data.get("id"),
                task_data.get("type"),
                task_data.get("status", "pending"),
                (
                    json.dumps(task_data.get("parameters"))
                    if task_data.get("parameters")
                    else None
                ),
                (
                    task_data.get("created_at").isoformat()
                    if task_data.get("created_at")
                    else None
                ),
                (
                    task_data.get("started_at").isoformat()
                    if task_data.get("started_at")
                    else None
                ),
                (
                    task_data.get("completed_at").isoformat()
                    if task_data.get("completed_at")
                    else None
                ),
                (
                    json.dumps(task_data.get("result"))
                    if task_data.get("result")
                    else None
                ),
                task_data.get("error"),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Saved task: {task_data.get('id')}")

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve task data by ID.

        Args:
            task_id: ID of the task

        Returns:
            Dictionary containing task information or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, type, status, parameters, created_at, started_at, completed_at, result, error
            FROM tasks WHERE id = ?
        """,
            (task_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "type": row[1],
                "status": row[2],
                "parameters": json.loads(row[3]) if row[3] else {},
                "created_at": datetime.fromisoformat(row[4]) if row[4] else None,
                "started_at": datetime.fromisoformat(row[5]) if row[5] else None,
                "completed_at": datetime.fromisoformat(row[6]) if row[6] else None,
                "result": json.loads(row[7]) if row[7] else {},
                "error": row[8],
            }

        return None

    def get_tasks(
        self, task_type: str = None, status: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve tasks with optional filters.

        Args:
            task_type: Type of task to filter by
            status: Status to filter by

        Returns:
            List of task dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT id, type, status, parameters, created_at, started_at, completed_at, result, error FROM tasks WHERE 1=1"
        params = []

        if task_type:
            query += " AND type = ?"
            params.append(task_type)

        if status:
            query += " AND status = ?"
            params.append(status)

        query += " ORDER BY created_at DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        tasks = []
        for row in rows:
            tasks.append(
                {
                    "id": row[0],
                    "type": row[1],
                    "status": row[2],
                    "parameters": json.loads(row[3]) if row[3] else {},
                    "created_at": datetime.fromisoformat(row[4]) if row[4] else None,
                    "started_at": datetime.fromisoformat(row[5]) if row[5] else None,
                    "completed_at": datetime.fromisoformat(row[6]) if row[6] else None,
                    "result": json.loads(row[7]) if row[7] else {},
                    "error": row[8],
                }
            )

        return tasks

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve symbol information by symbol code.

        Args:
            symbol: Symbol code to look up

        Returns:
            Dictionary containing symbol information or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT symbol, name, exchange, currency, sector, industry, created_at, updated_at
            FROM symbols WHERE symbol = ?
        """,
            (symbol,),
        )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "symbol": row[0],
                "name": row[1],
                "exchange": row[2],
                "currency": row[3],
                "sector": row[4],
                "industry": row[5],
                "created_at": datetime.fromisoformat(row[6]) if row[6] else None,
                "updated_at": datetime.fromisoformat(row[7]) if row[7] else None,
            }

        return None

    def get_symbols_by_filter(
        self, exchange: str = None, sector: str = None, industry: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve symbols with optional filters.

        Args:
            exchange: Exchange to filter by
            sector: Sector to filter by
            industry: Industry to filter by

        Returns:
            List of symbol dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT symbol, name, exchange, currency, sector, industry, created_at, updated_at FROM symbols WHERE 1=1"
        params = []

        if exchange:
            query += " AND exchange = ?"
            params.append(exchange)

        if sector:
            query += " AND sector = ?"
            params.append(sector)

        if industry:
            query += " AND industry = ?"
            params.append(industry)

        query += " ORDER BY symbol"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        symbols = []
        for row in rows:
            symbols.append(
                {
                    "symbol": row[0],
                    "name": row[1],
                    "exchange": row[2],
                    "currency": row[3],
                    "sector": row[4],
                    "industry": row[5],
                    "created_at": datetime.fromisoformat(row[6]) if row[6] else None,
                    "updated_at": datetime.fromisoformat(row[7]) if row[7] else None,
                }
            )

        return symbols

    # ==================== Scan Records Methods ====================

    def save_scan_record(self, scan_data: Dict[str, Any]) -> str:
        """
        Save a scan record to the database.

        Args:
            scan_data: Dictionary containing scan information
                - id: scan ID (optional, will be generated if not provided)
                - scan_date: date of the scan
                - market: market scanned (e.g., 'csi300')
                - days_analyzed: number of days analyzed
                - total_stocks: total number of stocks scanned
                - bias_filter_min: minimum BIAS filter value
                - bias_filter_max: maximum BIAS filter value
                - results: list of stock results

        Returns:
            scan_id: The ID of the saved scan record
        """
        import uuid

        scan_id = scan_data.get("id", str(uuid.uuid4()))
        scan_date = scan_data.get("scan_date", datetime.now().isoformat())
        created_at = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Save scan record
        cursor.execute(
            """
            INSERT OR REPLACE INTO scan_records
            (id, scan_date, market, days_analyzed, total_stocks, bias_filter_min, bias_filter_max, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                scan_id,
                scan_date,
                scan_data.get("market", ""),
                scan_data.get("days_analyzed", 0),
                scan_data.get("total_stocks", 0),
                scan_data.get("bias_filter_min"),
                scan_data.get("bias_filter_max"),
                created_at,
            ),
        )

        # Save individual stock results
        results = scan_data.get("results", [])
        for rank, result in enumerate(results, 1):
            cursor.execute(
                """
                INSERT OR REPLACE INTO scan_stock_results
                (scan_id, symbol, name, close_price, daily_return, total_score, rating, action, risk_level,
                 bias_6, bias_12, bias_24, trend_score, momentum_score, volatility_score, capital_score, structure_score, rank, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    scan_id,
                    result.get("symbol", ""),
                    result.get("name", ""),
                    result.get("close", 0),
                    result.get("daily_return", 0),
                    result.get("total_score", 0),
                    result.get("rating", ""),
                    result.get("action", ""),
                    result.get("risk_level", ""),
                    result.get("bias_6"),
                    result.get("bias_12"),
                    result.get("bias_24"),
                    result.get("trend_score", 0),
                    result.get("momentum_score", 0),
                    result.get("volatility_score", 0),
                    result.get("capital_score", 0),
                    result.get("structure_score", 0),
                    rank,
                    created_at,
                ),
            )

        conn.commit()
        conn.close()

        logger.info(f"Saved scan record: {scan_id} with {len(results)} stocks")
        return scan_id

    def get_scan_record(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a scan record by ID.

        Args:
            scan_id: ID of the scan record

        Returns:
            Dictionary containing scan record information or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get scan record
        cursor.execute(
            """
            SELECT id, scan_date, market, days_analyzed, total_stocks, bias_filter_min, bias_filter_max, created_at
            FROM scan_records WHERE id = ?
        """,
            (scan_id,),
        )

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        scan_record = {
            "id": row[0],
            "scan_date": row[1],
            "market": row[2],
            "days_analyzed": row[3],
            "total_stocks": row[4],
            "bias_filter_min": row[5],
            "bias_filter_max": row[6],
            "created_at": row[7],
        }

        # Get stock results
        cursor.execute(
            """
            SELECT symbol, name, close_price, daily_return, total_score, rating, action, risk_level,
                   bias_6, bias_12, bias_24, trend_score, momentum_score, volatility_score, capital_score, structure_score, rank
            FROM scan_stock_results
            WHERE scan_id = ?
            ORDER BY rank
        """,
            (scan_id,),
        )

        results = []
        for row in cursor.fetchall():
            results.append(
                {
                    "symbol": row[0],
                    "name": row[1],
                    "close": row[2],
                    "daily_return": row[3],
                    "total_score": row[4],
                    "rating": row[5],
                    "action": row[6],
                    "risk_level": row[7],
                    "bias_6": row[8],
                    "bias_12": row[9],
                    "bias_24": row[10],
                    "trend_score": row[11],
                    "momentum_score": row[12],
                    "volatility_score": row[13],
                    "capital_score": row[14],
                    "structure_score": row[15],
                    "rank": row[16],
                }
            )

        scan_record["results"] = results
        conn.close()

        return scan_record

    def get_scan_history(
        self,
        market: str = None,
        start_date: str = None,
        end_date: str = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve scan history with optional filters.

        Args:
            market: Market to filter by
            start_date: Start date (ISO format)
            end_date: End date (ISO format)
            limit: Maximum number of records to return

        Returns:
            List of scan record dictionaries
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        query = "SELECT id, scan_date, market, days_analyzed, total_stocks, bias_filter_min, bias_filter_max, created_at FROM scan_records WHERE 1=1"
        params = []

        if market:
            query += " AND market = ?"
            params.append(market)

        if start_date:
            query += " AND scan_date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND scan_date <= ?"
            params.append(end_date)

        query += " ORDER BY scan_date DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()

        scans = []
        for row in rows:
            scans.append(
                {
                    "id": row[0],
                    "scan_date": row[1],
                    "market": row[2],
                    "days_analyzed": row[3],
                    "total_stocks": row[4],
                    "bias_filter_min": row[5],
                    "bias_filter_max": row[6],
                    "created_at": row[7],
                }
            )

        return scans

    def get_scan_performance_summary(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """
        Get performance summary for a scan.

        Args:
            scan_id: ID of the scan record

        Returns:
            Dictionary with performance statistics or None
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT AVG(day_1_return), AVG(day_3_return), AVG(day_5_return),
                   AVG(day_10_return), AVG(day_20_return), AVG(day_60_return)
            FROM scan_performance_tracking
            WHERE scan_id = ?
        """,
            (scan_id,),
        )

        row = cursor.fetchone()
        conn.close()

        if row and row[0] is not None:
            return {
                "avg_day_1_return": row[0],
                "avg_day_3_return": row[1],
                "avg_day_5_return": row[2],
                "avg_day_10_return": row[3],
                "avg_day_20_return": row[4],
                "avg_day_60_return": row[5],
            }

        return None

    def update_scan_performance(self, tracking_data: Dict[str, Any]):
        """
        Update performance tracking for a scanned stock.

        Args:
            tracking_data: Dictionary containing performance data
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        updated_at = datetime.now().isoformat()

        cursor.execute(
            """
            INSERT OR REPLACE INTO scan_performance_tracking
            (scan_id, symbol, scan_date, scan_price, day_1_return, day_3_return, day_5_return,
             day_10_return, day_20_return, day_60_return, max_return_20d, max_drawdown_20d, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                tracking_data.get("scan_id"),
                tracking_data.get("symbol"),
                tracking_data.get("scan_date"),
                tracking_data.get("scan_price"),
                tracking_data.get("day_1_return"),
                tracking_data.get("day_3_return"),
                tracking_data.get("day_5_return"),
                tracking_data.get("day_10_return"),
                tracking_data.get("day_20_return"),
                tracking_data.get("day_60_return"),
                tracking_data.get("max_return_20d"),
                tracking_data.get("max_drawdown_20d"),
                updated_at,
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Updated performance tracking for {tracking_data.get('symbol')} in scan {tracking_data.get('scan_id')}")

    def compare_scans(self, scan_id_1: str, scan_id_2: str) -> Dict[str, Any]:
        """
        Compare two scans and return common stocks and differences.

        Args:
            scan_id_1: First scan ID
            scan_id_2: Second scan ID

        Returns:
            Dictionary with comparison results
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get stocks from first scan
        cursor.execute(
            "SELECT symbol, total_score, rank FROM scan_stock_results WHERE scan_id = ? ORDER BY rank",
            (scan_id_1,),
        )
        stocks_1 = {row[0]: {"score": row[1], "rank": row[2]} for row in cursor.fetchall()}

        # Get stocks from second scan
        cursor.execute(
            "SELECT symbol, total_score, rank FROM scan_stock_results WHERE scan_id = ? ORDER BY rank",
            (scan_id_2,),
        )
        stocks_2 = {row[0]: {"score": row[1], "rank": row[2]} for row in cursor.fetchall()}

        conn.close()

        # Find common stocks
        common_symbols = set(stocks_1.keys()) & set(stocks_2.keys())
        only_in_1 = set(stocks_1.keys()) - set(stocks_2.keys())
        only_in_2 = set(stocks_2.keys()) - set(stocks_1.keys())

        # Analyze common stocks
        common_stocks = []
        for symbol in common_symbols:
            common_stocks.append(
                {
                    "symbol": symbol,
                    "scan_1_rank": stocks_1[symbol]["rank"],
                    "scan_2_rank": stocks_2[symbol]["rank"],
                    "scan_1_score": stocks_1[symbol]["score"],
                    "scan_2_score": stocks_2[symbol]["score"],
                    "score_change": stocks_2[symbol]["score"] - stocks_1[symbol]["score"],
                    "rank_change": stocks_1[symbol]["rank"] - stocks_2[symbol]["rank"],
                }
            )

        # Sort by rank improvement
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

    def create_portfolio_backtest(self, backtest_data: Dict[str, Any]) -> str:
        """
        Create a new portfolio backtest record.

        Args:
            backtest_data: Dictionary containing backtest information
                - id: backtest ID (optional)
                - scan_id: associated scan record ID
                - portfolio_name: name of the portfolio
                - initial_capital: initial capital amount
                - start_date: start date of the backtest
                - end_date: end date of the backtest (optional)
                - status: status (active/closed/pending)

        Returns:
            backtest_id: The ID of the created backtest record
        """
        import uuid

        backtest_id = backtest_data.get("id", str(uuid.uuid4()))
        created_at = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO portfolio_backtests
            (id, scan_id, portfolio_name, initial_capital, start_date, end_date, status,
             total_return, annualized_return, sharpe_ratio, max_drawdown, created_at, completed_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                backtest_id,
                backtest_data.get("scan_id"),
                backtest_data.get("portfolio_name", f"Portfolio_{backtest_id[:8]}"),
                backtest_data.get("initial_capital", 500000),
                backtest_data.get("start_date"),
                backtest_data.get("end_date"),
                backtest_data.get("status", "active"),
                backtest_data.get("total_return"),
                backtest_data.get("annualized_return"),
                backtest_data.get("sharpe_ratio"),
                backtest_data.get("max_drawdown"),
                created_at,
                backtest_data.get("completed_at"),
            ),
        )

        conn.commit()
        conn.close()

        logger.info(f"Created portfolio backtest: {backtest_id}")
        return backtest_id

    def add_portfolio_holding(self, holding_data: Dict[str, Any]) -> int:
        """
        Add a holding to a portfolio backtest.

        Args:
            holding_data: Dictionary containing holding information

        Returns:
            holding_id: The ID of the created holding record
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO portfolio_holdings
            (backtest_id, symbol, name, entry_date, entry_price, shares, weight, status, exit_date, exit_price, realized_return)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                holding_data.get("backtest_id"),
                holding_data.get("symbol"),
                holding_data.get("name"),
                holding_data.get("entry_date"),
                holding_data.get("entry_price"),
                holding_data.get("shares"),
                holding_data.get("weight"),
                holding_data.get("status", "holding"),
                holding_data.get("exit_date"),
                holding_data.get("exit_price"),
                holding_data.get("realized_return"),
            ),
        )

        holding_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return holding_id

    def update_holding_exit(self, holding_id: int, exit_data: Dict[str, Any]):
        """Update holding with exit information."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE portfolio_holdings
            SET exit_date = ?, exit_price = ?, realized_return = ?, status = 'closed'
            WHERE id = ?
        """,
            (
                exit_data.get("exit_date"),
                exit_data.get("exit_price"),
                exit_data.get("realized_return"),
                holding_id,
            ),
        )

        conn.commit()
        conn.close()

    def record_daily_value(self, value_data: Dict[str, Any]):
        """Record daily portfolio value."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO portfolio_daily_values
            (backtest_id, date, total_value, cash_value, market_value, daily_return)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                value_data.get("backtest_id"),
                value_data.get("date"),
                value_data.get("total_value"),
                value_data.get("cash_value"),
                value_data.get("market_value"),
                value_data.get("daily_return"),
            ),
        )

        conn.commit()
        conn.close()

    def get_portfolio_backtest(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio backtest by ID with holdings."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get backtest record
        cursor.execute(
            """
            SELECT id, scan_id, portfolio_name, initial_capital, start_date, end_date, status,
                   total_return, annualized_return, sharpe_ratio, max_drawdown, created_at, completed_at
            FROM portfolio_backtests WHERE id = ?
        """,
            (backtest_id,),
        )

        row = cursor.fetchone()
        if not row:
            conn.close()
            return None

        backtest = {
            "id": row[0],
            "scan_id": row[1],
            "portfolio_name": row[2],
            "initial_capital": row[3],
            "start_date": row[4],
            "end_date": row[5],
            "status": row[6],
            "total_return": row[7],
            "annualized_return": row[8],
            "sharpe_ratio": row[9],
            "max_drawdown": row[10],
            "created_at": row[11],
            "completed_at": row[12],
        }

        # Get holdings
        cursor.execute(
            """
            SELECT id, symbol, name, entry_date, entry_price, shares, weight, status,
                   exit_date, exit_price, realized_return
            FROM portfolio_holdings WHERE backtest_id = ?
        """,
            (backtest_id,),
        )

        holdings = []
        for row in cursor.fetchall():
            holdings.append(
                {
                    "id": row[0],
                    "symbol": row[1],
                    "name": row[2],
                    "entry_date": row[3],
                    "entry_price": row[4],
                    "shares": row[5],
                    "weight": row[6],
                    "status": row[7],
                    "exit_date": row[8],
                    "exit_price": row[9],
                    "realized_return": row[10],
                }
            )

        backtest["holdings"] = holdings

        # Get daily values
        cursor.execute(
            """
            SELECT date, total_value, cash_value, market_value, daily_return
            FROM portfolio_daily_values WHERE backtest_id = ? ORDER BY date
        """,
            (backtest_id,),
        )

        daily_values = []
        for row in cursor.fetchall():
            daily_values.append(
                {
                    "date": row[0],
                    "total_value": row[1],
                    "cash_value": row[2],
                    "market_value": row[3],
                    "daily_return": row[4],
                }
            )

        backtest["daily_values"] = daily_values
        conn.close()

        return backtest

    def get_active_portfolios(self) -> List[Dict[str, Any]]:
        """Get all active (non-closed) portfolio backtests."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, scan_id, portfolio_name, initial_capital, start_date, status
            FROM portfolio_backtests WHERE status = 'active' ORDER BY created_at DESC
        """
        )

        portfolios = []
        for row in cursor.fetchall():
            portfolios.append(
                {
                    "id": row[0],
                    "scan_id": row[1],
                    "portfolio_name": row[2],
                    "initial_capital": row[3],
                    "start_date": row[4],
                    "status": row[5],
                }
            )

        conn.close()
        return portfolios

    def close_portfolio_backtest(self, backtest_id: str, metrics: Dict[str, Any]):
        """Close a portfolio backtest and update metrics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        completed_at = datetime.now().isoformat()

        cursor.execute(
            """
            UPDATE portfolio_backtests
            SET status = 'closed', end_date = ?, total_return = ?, annualized_return = ?,
                sharpe_ratio = ?, max_drawdown = ?, completed_at = ?
            WHERE id = ?
        """,
            (
                metrics.get("end_date"),
                metrics.get("total_return"),
                metrics.get("annualized_return"),
                metrics.get("sharpe_ratio"),
                metrics.get("max_drawdown"),
                completed_at,
                backtest_id,
            ),
        )

        conn.commit()
        conn.close()

    # ==================== Email Config Methods ====================

    def save_email_config(self, config_data: Dict[str, Any]) -> str:
        """Save email configuration."""
        import uuid

        config_id = config_data.get("id", str(uuid.uuid4()))

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT OR REPLACE INTO email_configs
            (id, smtp_host, smtp_port, username, password, from_addr, to_addrs, enabled)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                config_id,
                config_data.get("smtp_host"),
                config_data.get("smtp_port"),
                config_data.get("username"),
                config_data.get("password"),
                config_data.get("from_addr"),
                json.dumps(config_data.get("to_addrs", [])),
                config_data.get("enabled", True),
            ),
        )

        conn.commit()
        conn.close()

        return config_id

    def get_email_config(self, config_id: str = None) -> Optional[Dict[str, Any]]:
        """Get email configuration."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if config_id:
            cursor.execute(
                """
                SELECT id, smtp_host, smtp_port, username, password, from_addr, to_addrs, enabled
                FROM email_configs WHERE id = ?
            """,
                (config_id,),
            )
        else:
            cursor.execute(
                """
                SELECT id, smtp_host, smtp_port, username, password, from_addr, to_addrs, enabled
                FROM email_configs WHERE enabled = 1 LIMIT 1
            """
            )

        row = cursor.fetchone()
        conn.close()

        if row:
            return {
                "id": row[0],
                "smtp_host": row[1],
                "smtp_port": row[2],
                "username": row[3],
                "password": row[4],
                "from_addr": row[5],
                "to_addrs": json.loads(row[6]) if row[6] else [],
                "enabled": row[7],
            }

        return None

    # ==================== Scheduled Task Methods ====================

    def record_task_execution(self, task_data: Dict[str, Any]) -> str:
        """Record a scheduled task execution."""
        import uuid

        execution_id = task_data.get("id", str(uuid.uuid4()))
        executed_at = datetime.now().isoformat()

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO scheduled_tasks
            (id, task_type, schedule_expr, executed_at, status, result, error)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                execution_id,
                task_data.get("task_type"),
                task_data.get("schedule_expr"),
                executed_at,
                task_data.get("status", "success"),
                json.dumps(task_data.get("result")) if task_data.get("result") else None,
                task_data.get("error"),
            ),
        )

        conn.commit()
        conn.close()

        return execution_id

    def get_recent_task_executions(self, task_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task executions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        if task_type:
            cursor.execute(
                """
                SELECT id, task_type, schedule_expr, executed_at, status, result, error
                FROM scheduled_tasks WHERE task_type = ? ORDER BY executed_at DESC LIMIT ?
            """,
                (task_type, limit),
            )
        else:
            cursor.execute(
                """
                SELECT id, task_type, schedule_expr, executed_at, status, result, error
                FROM scheduled_tasks ORDER BY executed_at DESC LIMIT ?
            """,
                (limit,),
            )

        executions = []
        for row in cursor.fetchall():
            executions.append(
                {
                    "id": row[0],
                    "task_type": row[1],
                    "schedule_expr": row[2],
                    "executed_at": row[3],
                    "status": row[4],
                    "result": json.loads(row[5]) if row[5] else None,
                    "error": row[6],
                }
            )

        conn.close()
        return executions
