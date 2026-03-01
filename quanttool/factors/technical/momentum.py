"""Technical factor implementations."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ...domain.interfaces.factor import IFactor
from ...core.registry import registry, ComponentType
from ...core.logging import get_logger


logger = get_logger(__name__)


@registry.register(ComponentType.FACTOR, "momentum")
class MomentumFactor(IFactor):
    """Momentum factor implementation."""

    def __init__(self):
        """Initialize the momentum factor."""
        self.period = 10
        self.parameters = {"period": 10}

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
        Compute the momentum factor values based on input bars.

        Args:
            bars: Input price data with columns ['open', 'high', 'low', 'close', 'volume', ...]

        Returns:
            DataFrame with factor values aligned to the input bars
        """
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()

        # Calculate momentum as percentage change over the specified period
        df["factor_value"] = df["close"] / df["close"].shift(self.period) - 1

        return df[["timestamp", "factor_value"]].copy()

    def get_name(self) -> str:
        """
        Get the name of the factor.

        Returns:
            Factor name
        """
        return "Momentum"

    def get_description(self) -> str:
        """
        Get a description of the factor.

        Returns:
            Factor description
        """
        return f"Momentum factor calculated as percentage change over {self.period} periods"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the factor.

        Returns:
            Dictionary of parameters
        """
        return self.parameters.copy()


@registry.register(ComponentType.FACTOR, "returns_momentum")
class ReturnsMomentumFactor(IFactor):
    """Returns-based momentum factor implementation."""

    def __init__(self):
        """Initialize the returns momentum factor."""
        self.return_period = 5
        self.momentum_period = 20
        self.parameters = {"return_period": 5, "momentum_period": 20}

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the factor with parameters.

        Args:
            parameters: Factor-specific parameters
        """
        self.parameters.update(parameters)
        self.return_period = self.parameters["return_period"]
        self.momentum_period = self.parameters["momentum_period"]

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the returns momentum factor values based on input bars.

        Args:
            bars: Input price data with columns ['open', 'high', 'low', 'close', 'volume', ...]

        Returns:
            DataFrame with factor values aligned to the input bars
        """
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()

        # Calculate returns
        df["returns"] = df["close"].pct_change(periods=self.return_period)

        # Calculate momentum as average of returns over momentum period
        df["factor_value"] = df["returns"].rolling(window=self.momentum_period).mean()

        return df[["timestamp", "factor_value"]].copy()

    def get_name(self) -> str:
        """
        Get the name of the factor.

        Returns:
            Factor name
        """
        return "Returns_Momentum"

    def get_description(self) -> str:
        """
        Get a description of the factor.

        Returns:
            Factor description
        """
        return f"Returns momentum factor calculated as average of {self.return_period}-period returns over {self.momentum_period} periods"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the factor.

        Returns:
            Dictionary of parameters
        """
        return self.parameters.copy()


@registry.register(ComponentType.FACTOR, "price_volume_trend")
class PriceVolumeTrendFactor(IFactor):
    """Price Volume Trend factor implementation."""

    def __init__(self):
        """Initialize the PVT factor."""
        self.parameters = {}

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the factor with parameters.

        Args:
            parameters: Factor-specific parameters
        """
        self.parameters.update(parameters)

    def compute(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Compute the Price Volume Trend factor values based on input bars.

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

        # Calculate volume-adjusted returns (PVT-like measure)
        df["pvt_component"] = returns * df["volume"] / 1000000  # Normalize volume

        # Calculate cumulative sum to get the trend
        df["factor_value"] = df["pvt_component"].fillna(0).cumsum()

        return df[["timestamp", "factor_value"]].copy()

    def get_name(self) -> str:
        """
        Get the name of the factor.

        Returns:
            Factor name
        """
        return "Price_Volume_Trend"

    def get_description(self) -> str:
        """
        Get a description of the factor.

        Returns:
            Factor description
        """
        return "Price Volume Trend factor measuring cumulative volume-weighted price changes"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the factor.

        Returns:
            Dictionary of parameters
        """
        return self.parameters.copy()
