"""海龟交易策略实现 - 基于价格突破的唐奇安通道"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)


@registry.register(ComponentType.STRATEGY, "turtle")
class TurtleStrategy(IStrategy):
    """海龟交易策略

    经典的趋势跟踪策略，使用唐奇安通道（Donchian Channel）
    - 买入信号：价格突破N日最高价
    - 卖出信号：价格跌破M日最低价
    """

    def __init__(self):
        self.entry_period = 20  # 入场突破周期
        self.exit_period = 10   # 出场突破周期
        self.parameters = {
            "entry_period": 20,
            "exit_period": 10
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)
        self.entry_period = self.parameters["entry_period"]
        self.exit_period = self.parameters["exit_period"]
        logger.info(f"海龟策略初始化: entry={self.entry_period}, exit={self.exit_period}")

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()

        # 计算唐奇安通道
        df['upper_band'] = df['high'].rolling(window=self.entry_period).max().shift(1)
        df['lower_band'] = df['low'].rolling(window=self.entry_period).min().shift(1)
        df['exit_upper'] = df['high'].rolling(window=self.exit_period).max().shift(1)
        df['exit_lower'] = df['low'].rolling(window=self.exit_period).min().shift(1)

        df['signal'] = 0
        df['position'] = 0

        # 入场信号：突破上轨
        entry_signal = df['close'] > df['upper_band']

        # 出场信号：跌破下轨
        exit_signal = df['close'] < df['exit_lower']

        # 设置信号
        df.loc[entry_signal, 'signal'] = 1
        df.loc[exit_signal, 'signal'] = -1

        # 持仓状态
        position = 0
        positions = []
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1:
                position = 1
            elif df['signal'].iloc[i] == -1:
                position = 0
            positions.append(position)
        df['position'] = positions

        return df[['timestamp', 'signal', 'position', 'upper_band', 'lower_band']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        if len(historical_bars) < self.entry_period + 1:
            return {"direction": "hold", "reason": "insufficient_data"}

        upper_band = historical_bars['high'].rolling(window=self.entry_period).max().iloc[-2]
        lower_band = historical_bars['low'].rolling(window=self.entry_period).min().iloc[-2]
        exit_lower = historical_bars['low'].rolling(window=self.exit_period).min().iloc[-2]

        curr_close = current_bar['close']

        # 入场信号
        if curr_close > upper_band:
            return {
                "direction": "buy",
                "reason": "turtle_entry_breakout",
                "strength": 1.0,
                "upper_band": upper_band,
                "lower_band": lower_band
            }

        # 出场信号
        elif curr_close < exit_lower:
            return {
                "direction": "sell",
                "reason": "turtle_exit_breakout",
                "strength": 1.0,
                "exit_level": exit_lower
            }

        else:
            return {
                "direction": "hold",
                "reason": "no_breakout",
                "strength": 0.0,
                "upper_band": upper_band,
                "lower_band": lower_band
            }

    def get_name(self) -> str:
        return "Turtle"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return f"海龟策略: 入场{self.entry_period}日, 出场{self.exit_period}日"