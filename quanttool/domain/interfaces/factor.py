"""Abstract interface for factors in QuantTool."""

from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class IFactor(ABC):
    """Abstract interface for alpha factors."""

    @abstractmethod
    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the factor values based on input bars.

        Args:
            bars: Input price data with columns ['open', 'high', 'low', 'close', 'volume', ...]

        Returns:
            DataFrame with factor values aligned to the input bars
        """
        pass

    @abstractmethod
    def get_name(self) -> str:
        """
        Get the name of the factor.

        Returns:
            Factor name
        """
        pass

    @abstractmethod
    def get_description(self) -> str:
        """
        Get a description of the factor.

        Returns:
            Factor description
        """
        pass

    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the factor.

        Returns:
            Dictionary of parameters
        """
        pass
