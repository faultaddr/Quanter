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
