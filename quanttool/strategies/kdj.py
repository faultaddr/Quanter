"""KDJ策略实现 - 基于KDJ金叉死叉信号"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)


@registry.register(ComponentType.STRATEGY, "kdj")
class KDJStrategy(IStrategy):
    """KDJ策略

    买入信号：K线上穿D线（金叉）且J值<20
    卖出信号：K线下穿D线（死叉）且J值>80
    """

    def __init__(self):
        self.n = 9
        self.m1 = 3
        self.m2 = 3
        self.oversold = 20
        self.overbought = 80
        self.parameters = {
            "n": 9,
            "m1": 3,
            "m2": 3,
            "oversold": 20,
            "overbought": 80
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)
        self.n = self.parameters["n"]
        self.m1 = self.parameters["m1"]
        self.m2 = self.parameters["m2"]
        self.oversold = self.parameters["oversold"]
        self.overbought = self.parameters["overbought"]
        logger.info(f"KDJ策略初始化: n={self.n}, m1={self.m1}, m2={self.m2}")

    def _calculate_kdj(self, high: pd.Series, low: pd.Series, close: pd.Series,
                       n: int, m1: int, m2: int) -> tuple:
        """计算KDJ指标"""
        lowest_low = low.rolling(window=n).min()
        highest_high = high.rolling(window=n).max()

        rsv = (close - lowest_low) / (highest_high - lowest_low + 1e-10) * 100

        k = rsv.ewm(alpha=1/m1, adjust=False).mean()
        d = k.ewm(alpha=1/m2, adjust=False).mean()
        j = 3 * k - 2 * d

        return k, d, j

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()
        df['k'], df['d'], df['j'] = self._calculate_kdj(
            df['high'], df['low'], df['close'], self.n, self.m1, self.m2
        )

        df['signal'] = 0
        df['position'] = 0

        # 金叉买入：K上穿D且J<20
        golden_cross = (df['k'].shift(1) < df['d'].shift(1)) & (df['k'] >= df['d']) & (df['j'] < self.oversold + 10)

        # 死叉卖出：K下穿D且J>80
        death_cross = (df['k'].shift(1) > df['d'].shift(1)) & (df['k'] <= df['d']) & (df['j'] > self.overbought - 10)

        df.loc[golden_cross, 'signal'] = 1
        df.loc[death_cross, 'signal'] = -1

        df['position'] = df['signal'].cumsum().clip(lower=0, upper=1)

        return df[['timestamp', 'signal', 'position', 'k', 'd', 'j']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        if len(historical_bars) < self.n + 1:
            return {"direction": "hold", "reason": "insufficient_data"}

        k, d, j = self._calculate_kdj(
            historical_bars['high'], historical_bars['low'], historical_bars['close'],
            self.n, self.m1, self.m2
        )

        if len(k) < 2:
            return {"direction": "hold", "reason": "insufficient_kdj_data"}

        curr_k, curr_d, curr_j = k.iloc[-1], d.iloc[-1], j.iloc[-1]
        prev_k, prev_d = k.iloc[-2], d.iloc[-2]

        # 金叉买入
        if prev_k < prev_d and curr_k >= curr_d and curr_j < self.oversold + 10:
            return {
                "direction": "buy",
                "reason": "kdj_golden_cross_oversold",
                "strength": 1.0,
                "k": curr_k, "d": curr_d, "j": curr_j
            }

        # 死叉卖出
        elif prev_k > prev_d and curr_k <= curr_d and curr_j > self.overbought - 10:
            return {
                "direction": "sell",
                "reason": "kdj_death_cross_overbought",
                "strength": 1.0,
                "k": curr_k, "d": curr_d, "j": curr_j
            }

        else:
            return {
                "direction": "hold",
                "reason": "no_signal",
                "strength": 0.0,
                "k": curr_k, "d": curr_d, "j": curr_j
            }

    def get_name(self) -> str:
        return "KDJ"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return f"KDJ策略: 周期{self.n}, 超卖{self.oversold}, 超买{self.overbought}"