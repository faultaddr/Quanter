"""Database configuration for PostgreSQL."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """PostgreSQL database configuration."""

    host: str = "localhost"
    port: int = 5432
    database: str = "quanttool"
    user: str = "postgres"
    password: str = ""

    # Connection pool settings
    min_pool_size: int = 5
    max_pool_size: int = 20
    connection_timeout: float = 30.0
    command_timeout: float = 60.0

    # Retry settings
    max_retries: int = 3
    retry_delay: float = 1.0

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """
        Create configuration from environment variables.

        Environment variables:
            DB_HOST: Database host (default: localhost)
            DB_PORT: Database port (default: 5432)
            DB_NAME: Database name (default: quanttool)
            DB_USER: Database user (default: postgres)
            DB_PASSWORD: Database password
            DB_MIN_POOL_SIZE: Minimum pool size (default: 5)
            DB_MAX_POOL_SIZE: Maximum pool size (default: 20)
            DB_CONNECTION_TIMEOUT: Connection timeout in seconds (default: 30)
            DB_COMMAND_TIMEOUT: Command timeout in seconds (default: 60)
        """
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            database=os.getenv("DB_NAME", "quanttool"),
            user=os.getenv("DB_USER", "postgres"),
            password=os.getenv("DB_PASSWORD", ""),
            min_pool_size=int(os.getenv("DB_MIN_POOL_SIZE", "5")),
            max_pool_size=int(os.getenv("DB_MAX_POOL_SIZE", "20")),
            connection_timeout=float(os.getenv("DB_CONNECTION_TIMEOUT", "30.0")),
            command_timeout=float(os.getenv("DB_COMMAND_TIMEOUT", "60.0")),
        )

    @property
    def dsn(self) -> str:
        """Get the DSN (Data Source Name) for connection."""
        if self.password:
            return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}"

    @property
    def async_dsn(self) -> str:
        """Get the async DSN for asyncpg connection."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}" if self.password else f"postgresql://{self.user}@{self.host}:{self.port}/{self.database}"


# Default configuration instance
_config: Optional[DatabaseConfig] = None


def get_database_config() -> DatabaseConfig:
    """Get the database configuration (singleton)."""
    global _config
    if _config is None:
        _config = DatabaseConfig.from_env()
    return _config


def set_database_config(config: DatabaseConfig) -> None:
    """Set the database configuration."""
    global _config
    _config = config
