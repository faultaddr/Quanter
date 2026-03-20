"""Moving Average Cross strategy implementation."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger


logger = get_logger(__name__)


@registry.register(ComponentType.STRATEGY, "ma_cross")
class MACrossStrategy(IStrategy):
    """Moving Average Cross strategy implementation."""

    def __init__(self):
        """Initialize the strategy."""
        self.short_window = 10
        self.long_window = 30
        self.parameters = {"short_window": 10, "long_window": 30}

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the strategy with parameters.

        Args:
            parameters: Strategy-specific parameters
        """
        self.parameters.update(parameters)

        # Validate parameters
        if self.parameters["short_window"] >= self.parameters["long_window"]:
            raise ValueError("Short window must be less than long window")

        self.short_window = self.parameters["short_window"]
        self.long_window = self.parameters["long_window"]

        logger.info(
            f"MA Cross strategy initialized with short={self.short_window}, long={self.long_window}"
        )

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals based on input bars.

        Args:
            bars: Input price data with columns ['open', 'high', 'low', 'close', 'volume', ...]

        Returns:
            DataFrame with signal information
        """
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        # 需要足够的数据来计算移动平均线
        if len(bars) < self.long_window:
            # 返回空信号数据框
            result = pd.DataFrame({
                'timestamp': bars['timestamp'] if 'timestamp' in bars.columns else bars.index,
                'signal': [0] * len(bars),
                'position': [0] * len(bars)
            })
            return result

        df = bars.copy()

        # Calculate moving averages
        df[f"MA_{self.short_window}"] = (
            df["close"].rolling(window=self.short_window).mean()
        )
        df[f"MA_{self.long_window}"] = (
            df["close"].rolling(window=self.long_window).mean()
        )

        # Calculate signals
        df["position"] = 0
        df["position"][self.long_window :] = np.where(
            df[f"MA_{self.short_window}"][self.long_window :].values
            > df[f"MA_{self.long_window}"][self.long_window :].values,
            1,  # Buy signal
            0,  # Hold signal
        )

        # Calculate when to sell (when short MA crosses below long MA)
        df["position"][self.long_window :] = np.where(
            (
                df[f"MA_{self.short_window}"][self.long_window :].shift(1).values
                > df[f"MA_{self.long_window}"][self.long_window :].shift(1).values
            )
            & (
                df[f"MA_{self.short_window}"][self.long_window :].values
                <= df[f"MA_{self.long_window}"][self.long_window :].values
            ),
            -1,  # Sell signal
            df["position"][self.long_window :],
        )

        # Create signal column based on position changes
        df["signal"] = df["position"].diff()
        df["signal"] = df["signal"].fillna(0)

        return df[["timestamp", "signal", "position"]].copy()

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
        if len(historical_bars) < self.long_window:
            return {"direction": "hold", "reason": "insufficient_data"}

        # Get the last 'long_window' worth of bars to calculate moving averages
        recent_bars = historical_bars.tail(self.long_window + 10)  # Add buffer

        # Calculate moving averages for the recent data
        short_ma = recent_bars["close"].rolling(window=self.short_window).mean()
        long_ma = recent_bars["close"].rolling(window=self.long_window).mean()

        # Get the last few values to detect crossovers
        if len(short_ma) < 2 or len(long_ma) < 2:
            return {"direction": "hold", "reason": "insufficient_ma_data"}

        # Check for crossover
        current_short_ma = short_ma.iloc[-1]
        current_long_ma = long_ma.iloc[-1]
        prev_short_ma = short_ma.iloc[-2]
        prev_long_ma = long_ma.iloc[-2]

        # Golden cross (buy signal): short MA crosses above long MA
        if prev_short_ma <= prev_long_ma and current_short_ma > current_long_ma:
            return {"direction": "buy", "reason": "golden_cross", "strength": 1.0}

        # Death cross (sell signal): short MA crosses below long MA
        elif prev_short_ma >= prev_long_ma and current_short_ma < current_long_ma:
            return {"direction": "sell", "reason": "death_cross", "strength": 1.0}

        # Otherwise hold
        else:
            return {"direction": "hold", "reason": "no_cross", "strength": 0.0}

    def get_name(self) -> str:
        """
        Get the name of the strategy.

        Returns:
            Strategy name
        """
        return "MA_Cross"

    def get_parameters(self) -> Dict[str, Any]:
        """
        Get the current parameters of the strategy.

        Returns:
            Dictionary of parameters
        """
        return self.parameters.copy()

    def get_description(self) -> str:
        """
        Get a description of the strategy.

        Returns:
            Strategy description
        """
        return f"Moving Average Cross Strategy with short window {self.short_window} and long window {self.long_window}"
