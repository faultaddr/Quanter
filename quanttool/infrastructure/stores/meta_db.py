"""Metadata database for QuantTool.

This module now uses PostgreSQL instead of SQLite for better concurrency.
The interface remains the same for backward compatibility.
"""

# Re-export from the synchronous wrapper
from .meta_db_sync import MetaDB, get_meta_db

__all__ = ['MetaDB', 'get_meta_db']

# Log the migration
from ...core.logging import get_logger
logger = get_logger(__name__)
logger.info("MetaDB using PostgreSQL backend")
