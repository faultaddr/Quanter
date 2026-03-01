"""双均线策略实现 - 双MA交叉信号"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)


@registry.register(ComponentType.STRATEGY, "dual_ma")
class DualMAStrategy(IStrategy):
    """双均线策略

    买入信号：短期均线上穿长期均线（金叉）
    卖出信号：短期均线下穿长期均线（死叉）

    支持多组均线组合
    """

    def __init__(self):
        self.short_period = 5
        self.long_period = 20
        self.parameters = {
            "short_period": 5,
            "long_period": 20
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)
        self.short_period = self.parameters["short_period"]
        self.long_period = self.parameters["long_period"]

        if self.short_period >= self.long_period:
            raise ValueError("短期周期必须小于长期周期")

        logger.info(f"双均线策略初始化: short={self.short_period}, long={self.long_period}")

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()

        # 计算均线
        df['ma_short'] = df['close'].rolling(window=self.short_period).mean()
        df['ma_long'] = df['close'].rolling(window=self.long_period).mean()

        df['signal'] = 0
        df['position'] = 0

        # 金叉买入
        golden_cross = (df['ma_short'].shift(1) < df['ma_long'].shift(1)) & \
                       (df['ma_short'] >= df['ma_long'])

        # 死叉卖出
        death_cross = (df['ma_short'].shift(1) > df['ma_long'].shift(1)) & \
                      (df['ma_short'] <= df['ma_long'])

        df.loc[golden_cross, 'signal'] = 1
        df.loc[death_cross, 'signal'] = -1

        df['position'] = df['signal'].cumsum().clip(lower=0, upper=1)

        return df[['timestamp', 'signal', 'position', 'ma_short', 'ma_long']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        if len(historical_bars) < self.long_period + 1:
            return {"direction": "hold", "reason": "insufficient_data"}

        ma_short = historical_bars['close'].rolling(window=self.short_period).mean()
        ma_long = historical_bars['close'].rolling(window=self.long_period).mean()

        if len(ma_short) < 2:
            return {"direction": "hold", "reason": "insufficient_ma_data"}

        curr_short, curr_long = ma_short.iloc[-1], ma_long.iloc[-1]
        prev_short, prev_long = ma_short.iloc[-2], ma_long.iloc[-2]

        # 金叉
        if prev_short < prev_long and curr_short >= curr_long:
            return {
                "direction": "buy",
                "reason": "golden_cross",
                "strength": 1.0,
                "ma_short": curr_short,
                "ma_long": curr_long
            }

        # 死叉
        elif prev_short > prev_long and curr_short <= curr_long:
            return {
                "direction": "sell",
                "reason": "death_cross",
                "strength": 1.0,
                "ma_short": curr_short,
                "ma_long": curr_long
            }

        else:
            return {
                "direction": "hold",
                "reason": "no_cross",
                "strength": 0.0,
                "ma_short": curr_short,
                "ma_long": curr_long
            }

    def get_name(self) -> str:
        return "DualMA"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return f"双均线策略: 短期{self.short_period}日, 长期{self.long_period}日"