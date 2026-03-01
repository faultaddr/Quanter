"""布林带策略实现 - 基于布林带突破信号"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)


@registry.register(ComponentType.STRATEGY, "bollinger")
class BollingerBandStrategy(IStrategy):
    """布林带策略

    买入信号：股价跌破下轨后回归
    卖出信号：股价突破上轨后回归
    """

    def __init__(self):
        self.period = 20
        self.std_dev = 2.0
        self.parameters = {
            "period": 20,
            "std_dev": 2.0
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)
        self.period = self.parameters["period"]
        self.std_dev = self.parameters["std_dev"]
        logger.info(f"布林带策略初始化: period={self.period}, std_dev={self.std_dev}")

    def _calculate_bollinger(self, close: pd.Series, period: int, std_dev: float) -> tuple:
        """计算布林带"""
        mid = close.rolling(window=period).mean()
        std = close.rolling(window=period).std()
        upper = mid + std_dev * std
        lower = mid - std_dev * std
        return upper, mid, lower

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()
        df['upper'], df['mid'], df['lower'] = self._calculate_bollinger(
            df['close'], self.period, self.std_dev
        )

        df['signal'] = 0
        df['position'] = 0

        # 买入信号：股价从下轨下方回归到下轨上方
        prev_below_lower = df['close'].shift(1) < df['lower'].shift(1)
        curr_above_lower = df['close'] >= df['lower']
        buy_signal = prev_below_lower & curr_above_lower

        # 卖出信号：股价从上轨上方回归到上轨下方
        prev_above_upper = df['close'].shift(1) > df['upper'].shift(1)
        curr_below_upper = df['close'] <= df['upper']
        sell_signal = prev_above_upper & curr_below_upper

        df.loc[buy_signal, 'signal'] = 1
        df.loc[sell_signal, 'signal'] = -1

        df['position'] = df['signal'].cumsum().clip(lower=0, upper=1)

        return df[['timestamp', 'signal', 'position', 'upper', 'mid', 'lower']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        if len(historical_bars) < self.period + 1:
            return {"direction": "hold", "reason": "insufficient_data"}

        upper, mid, lower = self._calculate_bollinger(
            historical_bars['close'], self.period, self.std_dev
        )

        if len(upper) < 2:
            return {"direction": "hold", "reason": "insufficient_bollinger_data"}

        curr_close = current_bar['close']
        curr_upper = upper.iloc[-1]
        curr_lower = lower.iloc[-1]
        prev_close = historical_bars['close'].iloc[-1]
        prev_upper = upper.iloc[-2]
        prev_lower = lower.iloc[-2]

        # 买入信号
        if prev_close < prev_lower and curr_close >= curr_lower:
            return {
                "direction": "buy",
                "reason": "bollinger_lower_bounce",
                "strength": 1.0,
                "upper": curr_upper,
                "mid": mid.iloc[-1],
                "lower": curr_lower
            }

        # 卖出信号
        elif prev_close > prev_upper and curr_close <= curr_upper:
            return {
                "direction": "sell",
                "reason": "bollinger_upper_reject",
                "strength": 1.0,
                "upper": curr_upper,
                "mid": mid.iloc[-1],
                "lower": curr_lower
            }

        else:
            return {
                "direction": "hold",
                "reason": "no_signal",
                "strength": 0.0,
                "upper": curr_upper,
                "mid": mid.iloc[-1],
                "lower": curr_lower
            }

    def get_name(self) -> str:
        return "Bollinger"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return f"布林带策略: 周期{self.period}, 标准差倍数{self.std_dev}"