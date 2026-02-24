"""Abstract interface for strategies in QuantTool."""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional
import pandas as pd


class IStrategy(ABC):
    """Abstract interface for trading strategies."""

    @abstractmethod
    def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the strategy with parameters.

        Args:
            parameters: Strategy-specific parameters
        """
        pass

    @abstractmethod
    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals based on input bars.

        Args:
            bars: Input price data with columns ['open', 'high', 'low', 'close', 'volume', ...]

        Returns:
            DataFrame with signal information (columns like 'signal', 'position_size', etc.)
        """
        pass

    @abstractmethod
    def get_signal(
        self, current_bar: pd.Series, historical_bars: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get a signal for the current bar based on historical data.

        Args:
            current_bar: Current bar data
            historical_bars: Historical bar data up to current point

        Returns:
            Signal dictionary with action ('buy', 'sell', 'hold') and additional metadata
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the strategy.

        Returns:
            Strategy name
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the strategy.

        Returns:
            Dictionary of parameters
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of the strategy.

        Returns:
            Strategy description
        """
        pass
