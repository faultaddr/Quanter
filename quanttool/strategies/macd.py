"""MACD策略实现 - 基于MACD金叉死叉信号"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)


@registry.register(ComponentType.STRATEGY, "macd")
class MACDStrategy(IStrategy):
    """MACD策略

    买入信号：DIF上穿DEA（金叉）
    卖出信号：DIF下穿DEA（死叉）
    """

    def __init__(self):
        self.fast_period = 12
        self.slow_period = 26
        self.signal_period = 9
        self.parameters = {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9
        }

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)
        self.fast_period = self.parameters["fast_period"]
        self.slow_period = self.parameters["slow_period"]
        self.signal_period = self.parameters["signal_period"]
        logger.info(f"MACD策略初始化: fast={self.fast_period}, slow={self.slow_period}, signal={self.signal_period}")

    def _calculate_macd(self, close: pd.Series, fast: int, slow: int, signal: int) -> tuple:
        """计算MACD指标"""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        dif = ema_fast - ema_slow
        dea = dif.ewm(span=signal, adjust=False).mean()
        macd = (dif - dea) * 2
        return dif, dea, macd

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()
        df['dif'], df['dea'], df['macd'] = self._calculate_macd(
            df['close'], self.fast_period, self.slow_period, self.signal_period
        )

        df['signal'] = 0
        df['position'] = 0

        # 金叉买入信号
        prev_dif_below = df['dif'].shift(1) < df['dea'].shift(1)
        curr_dif_above = df['dif'] >= df['dea']
        golden_cross = prev_dif_below & curr_dif_above

        # 死叉卖出信号
        prev_dif_above = df['dif'].shift(1) > df['dea'].shift(1)
        curr_dif_below = df['dif'] <= df['dea']
        death_cross = prev_dif_above & curr_dif_below

        df.loc[golden_cross, 'signal'] = 1
        df.loc[death_cross, 'signal'] = -1

        df['position'] = df['signal'].cumsum().clip(lower=0, upper=1)

        return df[['timestamp', 'signal', 'position', 'dif', 'dea', 'macd']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        if len(historical_bars) < self.slow_period + 1:
            return {"direction": "hold", "reason": "insufficient_data"}

        dif, dea, macd = self._calculate_macd(
            historical_bars['close'], self.fast_period, self.slow_period, self.signal_period
        )

        if len(dif) < 2:
            return {"direction": "hold", "reason": "insufficient_macd_data"}

        curr_dif = dif.iloc[-1]
        curr_dea = dea.iloc[-1]
        prev_dif = dif.iloc[-2]
        prev_dea = dea.iloc[-2]

        # 金叉
        if prev_dif < prev_dea and curr_dif >= curr_dea:
            return {
                "direction": "buy",
                "reason": "golden_cross",
                "strength": 1.0,
                "dif": curr_dif,
                "dea": curr_dea
            }

        # 死叉
        elif prev_dif > prev_dea and curr_dif <= curr_dea:
            return {
                "direction": "sell",
                "reason": "death_cross",
                "strength": 1.0,
                "dif": curr_dif,
                "dea": curr_dea
            }

        else:
            return {
                "direction": "hold",
                "reason": "no_cross",
                "strength": 0.0,
                "dif": curr_dif,
                "dea": curr_dea
            }

    def get_name(self) -> str:
        return "MACD"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return f"MACD策略: 快线{self.fast_period}, 慢线{self.slow_period}, 信号线{self.signal_period}"