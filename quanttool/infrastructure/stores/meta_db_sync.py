"""Synchronous wrapper for AsyncMetaDB.

This module provides a synchronous interface for the async MetaDB,
allowing existing synchronous code to use PostgreSQL without modification.
"""

from typing import Dict, Any, List, Optional

from .meta_db_async import AsyncMetaDB, get_async_meta_db
from ..database.config import DatabaseConfig
from ..database.sync_loop import run_async


class MetaDB:
    """
    Synchronous wrapper for AsyncMetaDB.

    This class provides the same interface as the original SQLite-based MetaDB,
    but uses PostgreSQL internally.

    Usage:
        db = MetaDB()
        runs = db.get_experiment_runs()
    """

    def __init__(self, db_path: str = None, config: Optional[DatabaseConfig] = None):
        """
        Initialize the synchronous MetaDB wrapper.

        Args:
            db_path: Ignored (kept for backward compatibility)
            config: Database configuration
        """
        self._config = config
        self._async_db = None

    @property
    def async_db(self) -> AsyncMetaDB:
        """Get the async database instance."""
        if self._async_db is None:
            self._async_db = get_async_meta_db(self._config)
        return self._async_db

    # ==================== Experiment Run Methods ====================

    def save_experiment_run(self, run_data: Dict[str, Any]) -> None:
        """Save experiment run data."""
        return run_async(self.async_db.save_experiment_run(run_data))

    def get_experiment_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve experiment run data by ID."""
        return run_async(self.async_db.get_experiment_run(run_id))

    def get_experiment_runs(
        self, run_type: str = None, status: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve experiment runs with optional filters."""
        return run_async(self.async_db.get_experiment_runs(run_type, status, limit))

    # ==================== Task Methods ====================

    def save_task(self, task_data: Dict[str, Any]) -> None:
        """Save task data."""
        return run_async(self.async_db.save_task(task_data))

    def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve task data by ID."""
        return run_async(self.async_db.get_task(task_id))

    def get_tasks(
        self, task_type: str = None, status: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve tasks with optional filters."""
        return run_async(self.async_db.get_tasks(task_type, status, limit))

    # ==================== Symbol Methods ====================

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Retrieve symbol information."""
        return run_async(self.async_db.get_symbol_info(symbol))

    def get_symbols_by_filter(
        self, industry: str = None, market: str = None, status: str = "active", limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """Retrieve symbols with optional filters."""
        return run_async(self.async_db.get_symbols_by_filter(industry, market, status, limit))

    def upsert_symbol(self, symbol_data: Dict[str, Any]) -> None:
        """Insert or update symbol information."""
        return run_async(self.async_db.upsert_symbol(symbol_data))

    # ==================== Scan Record Methods ====================

    def save_scan_record(self, scan_data: Dict[str, Any]) -> str:
        """Save a scan record."""
        return run_async(self.async_db.save_scan_record(scan_data))

    def get_scan_record(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a scan record by ID."""
        return run_async(self.async_db.get_scan_record(scan_id))

    def get_scan_history(
        self, scan_type: str = None, start_date: str = None,
        end_date: str = None, limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Retrieve scan history."""
        return run_async(self.async_db.get_scan_history(scan_type, start_date, end_date, limit))

    def compare_scans(self, scan_id_1: str, scan_id_2: str) -> Dict[str, Any]:
        """Compare two scans."""
        return run_async(self.async_db.compare_scans(scan_id_1, scan_id_2))

    # ==================== Portfolio Backtest Methods ====================

    def create_portfolio_backtest(self, backtest_data: Dict[str, Any]) -> str:
        """Create a new portfolio backtest record."""
        return run_async(self.async_db.create_portfolio_backtest(backtest_data))

    def add_portfolio_holding(self, holding_data: Dict[str, Any]) -> int:
        """Add a holding to a portfolio backtest."""
        return run_async(self.async_db.add_portfolio_holding(holding_data))

    def update_holding_exit(self, holding_id: int, exit_data: Dict[str, Any]) -> None:
        """Update holding with exit information."""
        return run_async(self.async_db.update_holding_exit(holding_id, exit_data))

    def record_daily_value(self, value_data: Dict[str, Any]) -> None:
        """Record daily portfolio value."""
        return run_async(self.async_db.record_daily_value(value_data))

    def get_portfolio_backtest(self, backtest_id: str) -> Optional[Dict[str, Any]]:
        """Get portfolio backtest by ID."""
        return run_async(self.async_db.get_portfolio_backtest(backtest_id))

    def get_active_portfolios(self) -> List[Dict[str, Any]]:
        """Get all active portfolios."""
        return run_async(self.async_db.get_active_portfolios())

    def close_portfolio_backtest(self, backtest_id: str, metrics: Dict[str, Any]) -> None:
        """Close a portfolio backtest."""
        return run_async(self.async_db.close_portfolio_backtest(backtest_id, metrics))

    # ==================== Email Config Methods ====================

    def save_email_config(self, config_data: Dict[str, Any]) -> str:
        """Save email configuration."""
        return run_async(self.async_db.save_email_config(config_data))

    def get_email_config(self, config_id: str = None, name: str = None) -> Optional[Dict[str, Any]]:
        """Get email configuration."""
        return run_async(self.async_db.get_email_config(config_id, name))

    # ==================== Backward Compatibility ====================

    def init_tables(self):
        """Initialize tables (no-op, handled by async version)."""
        pass

    def get_scan_performance_summary(self, scan_id: str) -> Optional[Dict[str, Any]]:
        """Get performance summary (not yet implemented in async version)."""
        return None

    def update_scan_performance(self, tracking_data: Dict[str, Any]) -> None:
        """Update performance tracking (not yet implemented in async version)."""
        pass

    def record_task_execution(self, task_data: Dict[str, Any]) -> str:
        """Record task execution (use save_task instead)."""
        self.save_task(task_data)
        return task_data.get("id", "")

    def get_recent_task_executions(self, task_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent task executions (use get_tasks instead)."""
        return self.get_tasks(task_type=task_type, limit=limit)


def get_meta_db() -> MetaDB:
    """Get the MetaDB instance."""
    return MetaDB()
