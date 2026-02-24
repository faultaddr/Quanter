"""Volatility factor implementations."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ...domain.interfaces.factor import IFactor
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger


logger = get_logger(__name__)


@registry.register(ComponentType.FACTOR, "volatility")
class VolatilityFactor(IFactor):
    """Volatility factor implementation."""

    def __init__(self):
        """Initialize the volatility factor."""
        self.period = 20
        self.parameters = {"period": 20}

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the factor with parameters.

        Args:
            parameters: Factor-specific parameters
        """
        self.parameters.update(parameters)
        self.period = self.parameters["period"]

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the volatility factor values based on input bars.

        Args:
            bars: Input price data with columns ['open', 'high', 'low', 'close', 'volume', ...]

        Returns:
            DataFrame with factor values aligned to the input bars
        """
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()

        # Calculate returns
        returns = df["close"].pct_change()

        # Calculate rolling volatility (std dev of returns)
        df["factor_value"] = returns.rolling(window=self.period).std() * np.sqrt(
            252
        )  # Annualize

        return df[["timestamp", "factor_value"]].copy()

    def get_name(self) -> str:
        """
        Get the name of the factor.

        Returns:
            Factor name
        """
        return "Volatility"

    def get_description(self) -> str:
        """
        Get a description of the factor.

        Returns:
            Factor description
        """
        return f"Volatility factor calculated as rolling {self.period}-period standard deviation of returns (annualized)"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the factor.

        Returns:
            Dictionary of parameters
        """
        return self.parameters.copy()


@registry.register(ComponentType.FACTOR, "realized_volatility")
class RealizedVolatilityFactor(IFactor):
    """Realized volatility factor implementation using high-low prices."""

    def __init__(self):
        """Initialize the realized volatility factor."""
        self.period = 20
        self.parameters = {"period": 20}

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the factor with parameters.

        Args:
            parameters: Factor-specific parameters
        """
        self.parameters.update(parameters)
        self.period = self.parameters["period"]

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the realized volatility factor values based on input bars.

        Args:
            bars: Input price data with columns ['open', 'high', 'low', 'close', 'volume', ...]

        Returns:
            DataFrame with factor values aligned to the input bars
        """
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()

        # Calculate Garman-Klass volatility estimator
        # Formula: 0.5 * log(H/L)^2 - (2*ln(2)-1) * log(C/O)^2
        df["log_hl"] = np.log(df["high"] / df["low"]) ** 2
        df["log_co"] = np.log(df["close"] / df["open"]) ** 2

        garman_klass = 0.5 * df["log_hl"] - (2 * np.log(2) - 1) * df["log_co"]

        # Calculate rolling realized volatility
        df["factor_value"] = np.sqrt(
            df["garman_klass"].rolling(window=self.period).mean()
        )

        return df[["timestamp", "factor_value"]].copy()

    def get_name(self) -> str:
        """
        Get the name of the factor.

        Returns:
            Factor name
        """
        return "Realized_Volatility"

    def get_description(self) -> str:
        """
        Get a description of the factor.

        Returns:
            Factor description
        """
        return f"Realized volatility factor calculated using Garman-Klass estimator over {self.period} periods"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the factor.

        Returns:
            Dictionary of parameters
        """
        return self.parameters.copy()
