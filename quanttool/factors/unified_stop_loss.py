"""
统一止损计算 - 单一止损价格来源

核心理念：
- 止损计算统一，避免多处不同止损逻辑
- 优先级：形态止损 > 支撑位止损 > ATR止损 > 均线止损

止损类型：
1. 形态止损：突破系统检测到有效形态，使用形态最低点
2. 支撑位止损：有明确支撑位（前低、盘整下沿）
3. ATR止损：基于平均真实波动范围
4. 均线止损：跌破重要均线
"""
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .analysis_context import (
    StopLossConfig,
    StopLossType,
    AnalysisContext,
    BreakoutScore,
)


class UnifiedStopLossCalculator:
    """
    统一止损计算器

    确保整个系统使用统一的止损逻辑
    """

    # ATR止损倍数
    ATR_MULTIPLIER = 2.0

    # 形态止损确认比例
    PATTERN_CONFIRM_RATIO = 0.98  # 形态最低点下方2%

    # 支撑位确认比例
    SUPPORT_CONFIRM_RATIO = 0.98  # 支撑位下方2%

    # 均线止损MA
    MA_STOP_PERIOD = 20  # MA20止损

    # 固定比例止损
    DEFAULT_STOP_PERCENT = 0.05  # 默认5%止损

    def calculate(
        self,
        df: pd.DataFrame,
        context: AnalysisContext,
        preferred_type: StopLossType = StopLossType.ATR
    ) -> StopLossConfig:
        """
        计算统一止损

        优先级：
        1. 形态止损（突破系统检测到有效形态）
        2. 支撑位止损（有明确支撑）
        3. ATR止损（默认）
        4. 均线止损

        Args:
            df: 股票数据
            context: 分析上下文
            preferred_type: 首选止损类型

        Returns:
            StopLossConfig: 止损配置
        """
        config = StopLossConfig()

        if df.empty:
            return config

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values
        current_price = close[-1]

        # 计算各类止损
        pattern_stop = self._calculate_pattern_stop(context)
        support_stop = self._calculate_support_stop(df)
        atr_stop, atr_value = self._calculate_atr_stop(df)
        ma_stop = self._calculate_ma_stop(df)

        # 存储各类止损价格
        config.pattern_stop = pattern_stop
        config.support_stop = support_stop
        config.atr_stop = atr_stop
        config.ma_stop = ma_stop
        config.atr_value = atr_value

        # 根据优先级选择止损
        # 1. 形态止损（如果有突破形态）
        if pattern_stop > 0 and context.breakout_score.has_breakout:
            config.stop_price = pattern_stop
            config.stop_type = StopLossType.PATTERN
            config.confidence = 0.9

        # 2. 支撑位止损（如果有明确支撑）
        elif support_stop > 0:
            config.stop_price = support_stop
            config.stop_type = StopLossType.SUPPORT
            config.confidence = 0.8

        # 3. ATR止损（默认）
        elif atr_stop > 0:
            config.stop_price = atr_stop
            config.stop_type = StopLossType.ATR
            config.confidence = 0.7

        # 4. 均线止损
        elif ma_stop > 0:
            config.stop_price = ma_stop
            config.stop_type = StopLossType.MA
            config.confidence = 0.6

        # 5. 固定比例止损（兜底）
        else:
            config.stop_price = current_price * (1 - self.DEFAULT_STOP_PERCENT)
            config.stop_type = StopLossType.PERCENTAGE
            config.confidence = 0.5

        # 计算止损幅度
        if current_price > 0:
            config.distance_percent = (current_price - config.stop_price) / current_price

        # 计算止盈价格
        risk = current_price - config.stop_price
        if risk > 0:
            config.take_profit_price = current_price + risk * config.risk_reward_ratio

        return config

    def _calculate_pattern_stop(self, context: AnalysisContext) -> float:
        """
        计算形态止损

        使用突破系统检测到的形态止损价格
        """
        breakout = context.breakout_score

        if not breakout.passed_filter:
            return 0.0

        # 使用突破系统的止损价格
        if breakout.stop_loss_price > 0:
            return breakout.stop_loss_price

        # 如果没有预计算的止损价格，使用盘整下沿
        if breakout.is_consolidating:
            details = breakout.details
            consolidation = details.get('consolidation', {})
            low = consolidation.get('low', 0)
            if low > 0:
                return low * self.PATTERN_CONFIRM_RATIO

        return 0.0

    def _calculate_support_stop(self, df: pd.DataFrame) -> float:
        """
        计算支撑位止损

        寻找近期有效支撑位：
        1. 前期低点
        2. 盘整区间下沿
        3. 跳空缺口支撑
        """
        if len(df) < 20:
            return 0.0

        close = df['close'].values
        low = df['low'].values
        high = df['high'].values

        # 寻找近期低点
        lookback = min(60, len(df))
        recent_low = np.min(low[-lookback:])

        # 寻找支撑位（多次测试的低点区域）
        support_levels = self._find_support_levels(df, lookback)

        if support_levels:
            # 使用最近的支撑位
            current_price = close[-1]
            valid_supports = [s for s in support_levels if s < current_price]
            if valid_supports:
                return max(valid_supports) * self.SUPPORT_CONFIRM_RATIO

        # 使用近期低点
        if recent_low > 0:
            return recent_low * self.SUPPORT_CONFIRM_RATIO

        return 0.0

    def _calculate_atr_stop(self, df: pd.DataFrame) -> Tuple[float, float]:
        """
        计算ATR止损

        Returns:
            Tuple[float, float]: (止损价格, ATR值)
        """
        if len(df) < 14:
            return 0.0, 0.0

        close = df['close'].values
        high = df['high'].values
        low = df['low'].values

        # 计算ATR
        atr = self._calculate_atr(df, 14)
        current_price = close[-1]

        if atr > 0 and current_price > 0:
            stop_price = current_price - atr * self.ATR_MULTIPLIER
            return stop_price, atr

        return 0.0, 0.0

    def _calculate_ma_stop(self, df: pd.DataFrame) -> float:
        """
        计算均线止损

        使用MA20作为止损位
        """
        if len(df) < self.MA_STOP_PERIOD:
            return 0.0

        close = df['close'].values
        ma = self._calculate_ma(close, self.MA_STOP_PERIOD)

        if len(ma) > 0 and not np.isnan(ma[-1]):
            return ma[-1]

        return 0.0

    def _find_support_levels(
        self,
        df: pd.DataFrame,
        lookback: int
    ) -> list:
        """
        寻找支撑位

        识别多次测试的低点区域
        """
        if len(df) < lookback:
            return []

        low = df['low'].values
        close = df['close'].values

        support_levels = []

        # 简单支撑位识别：局部低点
        for i in range(2, lookback - 2):
            idx = -i
            # 检查是否是局部低点
            if (low[idx] < low[idx-1] and low[idx] < low[idx+1] and
                low[idx] < low[idx-2] and low[idx] < low[idx+2]):
                support_levels.append(low[idx])

        # 去重并排序
        if support_levels:
            support_levels = list(set(support_levels))
            support_levels.sort(reverse=True)

        return support_levels[:3]  # 最多返回3个支撑位

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> float:
        """计算ATR指标"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values

        if len(close) < period + 1:
            return 0.0

        # 计算True Range
        tr = np.zeros(len(close))
        for i in range(1, len(close)):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i-1]),
                abs(low[i] - close[i-1])
            )

        # 计算ATR
        atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
        return atr

    def _calculate_ma(self, data: np.ndarray, period: int) -> np.ndarray:
        """计算移动平均线"""
        result = np.full(len(data), np.nan)
        if len(data) >= period:
            result[period-1:] = np.convolve(data, np.ones(period)/period, mode='valid')
        return result


def create_stop_loss_calculator() -> UnifiedStopLossCalculator:
    """创建止损计算器实例"""
    return UnifiedStopLossCalculator()