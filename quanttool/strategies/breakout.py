"""Breakout strategy implementation."""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger


logger = get_logger(__name__)


@registry.register(ComponentType.STRATEGY, "breakout")
class BreakoutStrategy(IStrategy):
    """Breakout strategy implementation."""

    def __init__(self):
        """Initialize the strategy."""
        self.lookback_period = 20
        self.entry_threshold = 0.02  # 2% above/below threshold
        self.parameters = {"lookback_period": 20, "entry_threshold": 0.02}

    def initialize(self, parameters: Dict[str, Any]) -> None:
        """
        Initialize the strategy with parameters.

        Args:
            parameters: Strategy-specific parameters
        """
        self.parameters.update(parameters)

        self.lookback_period = self.parameters["lookback_period"]
        self.entry_threshold = self.parameters["entry_threshold"]

        logger.info(
            f"Breakout strategy initialized with lookback={self.lookback_period}, threshold={self.entry_threshold}"
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

        df = bars.copy()

        # Calculate rolling max/min for breakout levels
        df["upper_band"] = df["high"].rolling(window=self.lookback_period).max()
        df["lower_band"] = df["low"].rolling(window=self.lookback_period).min()

        # Calculate signals based on breakouts
        df["signal"] = 0
        df["position"] = 0

        # Long signal: close above upper band + entry threshold
        df["long_signal"] = df["close"] > df["upper_band"].shift(1) * (
            1 + self.entry_threshold
        )

        # Short signal: close below lower band - entry threshold
        df["short_signal"] = df["close"] < df["lower_band"].shift(1) * (
            1 - self.entry_threshold
        )

        # Set position based on signals
        df.loc[df["long_signal"], "position"] = 1
        df.loc[df["short_signal"], "position"] = -1

        # Forward fill positions to maintain until exit signal
        df["position"] = df["position"].replace(to_replace=0, method="ffill").fillna(0)

        # Only trigger signal on actual breakouts (don't keep signaling every bar)
        df["signal"] = df["position"].diff()
        df["signal"] = df["signal"].fillna(0)

        return df[
            ["timestamp", "signal", "position", "upper_band", "lower_band"]
        ].copy()

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
        if len(historical_bars) < self.lookback_period:
            return {"direction": "hold", "reason": "insufficient_data"}

        # Get the last 'lookback_period' worth of bars to calculate breakout levels
        recent_bars = historical_bars.tail(self.lookback_period + 10)  # Add buffer

        # Calculate rolling max/min for breakout levels
        upper_band = recent_bars["high"].rolling(window=self.lookback_period).max()
        lower_band = recent_bars["low"].rolling(window=self.lookback_period).min()

        if len(upper_band) < 2 or len(lower_band) < 2:
            return {"direction": "hold", "reason": "insufficient_levels_data"}

        # Get current and previous levels
        current_upper = upper_band.iloc[-1]
        current_lower = lower_band.iloc[-1]
        prev_upper = upper_band.iloc[-2]
        prev_lower = lower_band.iloc[-2]

        current_close = current_bar["close"]

        # Long signal: close above upper band + entry threshold
        if current_close > prev_upper * (1 + self.entry_threshold):
            return {
                "direction": "buy",
                "reason": "upper_breakout",
                "strength": 1.0,
                "breakout_level": current_upper,
                "current_price": current_close,
            }

        # Short signal: close below lower band - entry threshold
        elif current_close < prev_lower * (1 - self.entry_threshold):
            return {
                "direction": "sell",
                "reason": "lower_breakout",
                "strength": 1.0,
                "breakout_level": current_lower,
                "current_price": current_close,
            }

        # Otherwise hold
        else:
            return {
                "direction": "hold",
                "reason": "no_breakout",
                "strength": 0.0,
                "upper_band": current_upper,
                "lower_band": current_lower,
                "current_price": current_close,
            }

    def get_name(self) -> str:
        """
        Get the name of the strategy.

        Returns:
            Strategy name
        """
        return "Breakout"

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
        return f"Breakout Strategy with lookback period {self.lookback_period} and entry threshold {self.entry_threshold*100}%"
