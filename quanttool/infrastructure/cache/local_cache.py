"""Local Data Cache for QuantTool.

This module now uses PostgreSQL instead of SQLite for better concurrency.
The interface remains the same for backward compatibility.
"""

# Re-export from the synchronous wrapper
from .local_cache_sync import LocalDataCache, get_local_cache

__all__ = ['LocalDataCache', 'get_local_cache']

# Log the migration
from ...core.logging import get_logger
logger = get_logger(__name__)
logger.info("LocalDataCache using PostgreSQL backend")
