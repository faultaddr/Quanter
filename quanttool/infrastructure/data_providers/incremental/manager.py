"""Incremental Data Manager for QuantTool.

This module now uses PostgreSQL instead of SQLite for better concurrency.
The interface remains the same for backward compatibility.
"""

# Re-export from the synchronous wrapper
from .sync_manager import (
    IncrementalDataManager,
    DataType,
    DataRange,
    get_incremental_manager,
)

__all__ = [
    'IncrementalDataManager',
    'DataType',
    'DataRange',
    'get_incremental_manager',
]

# Log the migration
from ....core.logging import get_logger
logger = get_logger(__name__)
logger.info("IncrementalDataManager using PostgreSQL backend")
