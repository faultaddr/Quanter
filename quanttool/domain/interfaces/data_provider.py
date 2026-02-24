"""Abstract interfaces for data providers in QuantTool."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Dict, Any
import pandas as pd
from ...core.errors import DataProviderError


class IDataProvider(ABC):
    """Abstract interface for data providers."""

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the data provider (e.g., connect to API, validate credentials)."""
        pass

    @abstractmethod
    def get_supported_symbols(self) -> List[str]:
        """Get list of supported symbols."""
        pass

    @abstractmethod
    def get_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
        timeframe: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Get OHLCV bars for the given symbols and timeframe.

        Args:
            symbols: List of symbols to retrieve
            start_date: Start date for data retrieval
            end_date: End date for data retrieval
            timeframe: Timeframe string (e.g., '1m', '5m', '10m', '1d')

        Returns:
            Dictionary mapping symbols to their OHLCV dataframes
        """
        pass

    @abstractmethod
    def get_latest_bar(
        self, symbol: str, timeframe: str = "10m"
    ) -> Optional[pd.DataFrame]:
        """
        Get the most recent bar for a symbol.

        Args:
            symbol: Symbol to retrieve
            timeframe: Timeframe string (e.g., '1m', '5m', '10m')

        Returns:
            DataFrame with the latest bar data, or None if unavailable
        """
        pass

    @abstractmethod
    def search_symbols(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for symbols matching the query.

        Args:
            query: Search query string

        Returns:
            List of matching symbols with metadata
        """
        pass

    @abstractmethod
    def get_calendar(self) -> List[datetime]:
        """
        Get trading calendar (list of trading days).

        Returns:
            List of trading days
        """
        pass
