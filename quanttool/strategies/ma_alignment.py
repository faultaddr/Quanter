"""均线多头排列策略 - 基于多均线系统"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from ..domain.interfaces.strategy import IStrategy
from ..core.registry import registry, ComponentType
from ..core.logging import get_logger

logger = get_logger(__name__)


@registry.register(ComponentType.STRATEGY, "ma_alignment")
class MAAlignmentStrategy(IStrategy):
    """均线多头排列策略

    多头排列：MA5 > MA10 > MA20 > MA60
    空头排列：MA5 < MA10 < MA20 < MA60

    买入信号：均线形成多头排列
    卖出信号：均线形成空头排列或跌破MA20
    """

    def __init__(self):
        self.parameters = {}

    def initialize(self, parameters: Dict[str, Any]) -> None:
        self.parameters.update(parameters)
        logger.info("均线排列策略初始化")

    def calculate_signals(self, bars: pd.DataFrame) -> pd.DataFrame:
        if bars is None or len(bars) == 0:
            return pd.DataFrame()

        df = bars.copy()

        # 计算各周期均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()

        df['signal'] = 0
        df['position'] = 0

        # 多头排列
        bullish = (df['ma5'] > df['ma10']) & (df['ma10'] > df['ma20']) & (df['ma20'] > df['ma60'])
        # 空头排列
        bearish = (df['ma5'] < df['ma10']) & (df['ma10'] < df['ma20']) & (df['ma20'] < df['ma60'])

        # 信号：多头排列转空头排列时卖出，空头转多头时买入
        df.loc[bullish & ~bullish.shift(1).fillna(False), 'signal'] = 1
        df.loc[bearish & ~bearish.shift(1).fillna(False), 'signal'] = -1

        # 持仓状态：多头排列持多仓
        df['position'] = bullish.astype(int)

        return df[['timestamp', 'signal', 'position', 'ma5', 'ma10', 'ma20', 'ma60']].copy()

    def get_signal(self, current_bar: pd.Series, historical_bars: pd.DataFrame) -> Dict[str, Any]:
        if len(historical_bars) < 60:
            return {"direction": "hold", "reason": "insufficient_data"}

        ma5 = historical_bars['close'].rolling(window=5).mean().iloc[-1]
        ma10 = historical_bars['close'].rolling(window=10).mean().iloc[-1]
        ma20 = historical_bars['close'].rolling(window=20).mean().iloc[-1]
        ma60 = historical_bars['close'].rolling(window=60).mean().iloc[-1]

        # 前一日均线
        prev_ma5 = historical_bars['close'].rolling(window=5).mean().iloc[-2]
        prev_ma10 = historical_bars['close'].rolling(window=10).mean().iloc[-2]
        prev_ma20 = historical_bars['close'].rolling(window=20).mean().iloc[-2]
        prev_ma60 = historical_bars['close'].rolling(window=60).mean().iloc[-2]

        # 判断排列
        curr_bullish = (ma5 > ma10) and (ma10 > ma20) and (ma20 > ma60)
        prev_bullish = (prev_ma5 > prev_ma10) and (prev_ma10 > prev_ma20) and (prev_ma20 > prev_ma60)
        curr_bearish = (ma5 < ma10) and (ma10 < ma20) and (ma20 < ma60)
        prev_bearish = (prev_ma5 < prev_ma10) and (prev_ma10 < prev_ma20) and (prev_ma20 < prev_ma60)

        # 多头排列形成
        if curr_bullish and not prev_bullish:
            return {
                "direction": "buy",
                "reason": "bullish_alignment_formed",
                "strength": 1.0,
                "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60
            }

        # 空头排列形成
        elif curr_bearish and not prev_bearish:
            return {
                "direction": "sell",
                "reason": "bearish_alignment_formed",
                "strength": 1.0,
                "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60
            }

        else:
            alignment = "bullish" if curr_bullish else "bearish" if curr_bearish else "neutral"
            return {
                "direction": "hold",
                "reason": f"ma_alignment_{alignment}",
                "strength": 0.0,
                "ma5": ma5, "ma10": ma10, "ma20": ma20, "ma60": ma60
            }

    def get_name(self) -> str:
        return "MA_Alignment"

    def get_parameters(self) -> Dict[str, Any]:
        return self.parameters.copy()

    def get_description(self) -> str:
        return "均线多头排列策略: MA5 > MA10 > MA20 > MA60 为多头，反之为空头"