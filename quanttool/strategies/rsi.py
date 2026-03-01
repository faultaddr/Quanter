"""RSI策略实现 - 基于RSI超买超卖信号"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)


@registry.register(ComponentType.STRATEGY, "rsi")
class RSIStrategy(IStrategy):
    """RSI超买超卖策略

    买入信号：RSI从超卖区（<30）上穿30
    卖出信号：RSI从超买区（>70）下穿70
    """

    def __init__(self):
        self.period = 14
        self.oversold = 30
        self.overbought = 70
        self.parameters = {
            "period": 14,
            "oversold": 30,
            "overbought": 70
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)
        self.period = self.parameters["period"]
        self.oversold = self.parameters["oversold"]
        self.overbought = self.parameters["overbought"]
        logger.info(f"RSI策略初始化: period={self.period}, oversold={self.oversold}, overbought={self.overbought}")

    def _calculate_rsi(self, close: pd.Series, period: int) -> pd.Series:
        """计算RSI指标"""
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()
        df['rsi'] = self._calculate_rsi(df['close'], self.period)

        df['signal'] = 0
        df['position'] = 0

        # 买入信号：RSI从超卖区上穿oversold
        df['prev_rsi'] = df['rsi'].shift(1)
        buy_signal = (df['prev_rsi'] < self.oversold) & (df['rsi'] >= self.oversold)

        # 卖出信号：RSI从超买区下穿overbought
        sell_signal = (df['prev_rsi'] > self.overbought) & (df['rsi'] <= self.overbought)

        df.loc[buy_signal, 'signal'] = 1
        df.loc[sell_signal, 'signal'] = -1

        # 计算持仓状态
        df['position'] = df['signal'].cumsum().clip(lower=0, upper=1)

        return df[['timestamp', 'signal', 'position', 'rsi']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        if len(historical_bars) < self.period + 1:
            return {"direction": "hold", "reason": "insufficient_data"}

        rsi_series = self._calculate_rsi(historical_bars['close'], self.period)

        if len(rsi_series) < 2:
            return {"direction": "hold", "reason": "insufficient_rsi_data"}

        current_rsi = rsi_series.iloc[-1]
        prev_rsi = rsi_series.iloc[-2]

        # 买入信号
        if prev_rsi < self.oversold and current_rsi >= self.oversold:
            return {
                "direction": "buy",
                "reason": "rsi_oversold_exit",
                "strength": 1.0,
                "rsi": current_rsi
            }

        # 卖出信号
        elif prev_rsi > self.overbought and current_rsi <= self.overbought:
            return {
                "direction": "sell",
                "reason": "rsi_overbought_exit",
                "strength": 1.0,
                "rsi": current_rsi
            }

        else:
            return {
                "direction": "hold",
                "reason": "no_signal",
                "strength": 0.0,
                "rsi": current_rsi
            }

    def get_name(self) -> str:
        return "RSI"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return f"RSI策略: 周期{self.period}, 超卖{self.oversold}, 超买{self.overbought}"