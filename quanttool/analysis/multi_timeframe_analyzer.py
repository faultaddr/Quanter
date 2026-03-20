"""
多周期共振分析模块

实现多周期信号确认：
- 周期数据获取（日/周/月）
- 跨周期信号一致性检查
- 对齐评分计算
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class TimeframeAlignment(str, Enum):
    """周期对齐状态"""
    FULLY_ALIGNED = "fully_aligned"      # 三周期完全对齐
    PARTIALLY_ALIGNED = "partially_aligned"  # 两周期对齐
    DIVERGENT = "divergent"               # 周期背离
    INSUFFICIENT_DATA = "insufficient_data"  # 数据不足


@dataclass
class TimeframeScore:
    """单周期评分"""
    timeframe: str
    score: float
    trend: str  # 'up', 'down', 'sideway'
    strength: float
    signal: str  # 'buy', 'sell', 'hold'


@dataclass
class MultiTimeframeResult:
    """多周期分析结果"""
    daily: TimeframeScore
    weekly: Optional[TimeframeScore]
    monthly: Optional[TimeframeScore]
    alignment: TimeframeAlignment
    combined_score: float
    alignment_bonus: float
    confidence: float


class MultiTimeframeAnalyzer:
    """
    多周期分析器

    检查跨周期信号一致性，计算多周期综合评分
    """

    # 时间周期定义
    TIMEFRAMES = {
        'daily': 1,
        'weekly': 5,
        'monthly': 20
    }

    # 权重配置
    TIMEFRAME_WEIGHTS = {
        'daily': 0.50,
        'weekly': 0.35,
        'monthly': 0.15
    }

    # 对齐奖励
    ALIGNMENT_BONUS = {
        TimeframeAlignment.FULLY_ALIGNED: 0.10,
        TimeframeAlignment.PARTIALLY_ALIGNED: 0.05,
        TimeframeAlignment.DIVERGENT: -0.10,
        TimeframeAlignment.INSUFFICIENT_DATA: 0.0
    }

    def __init__(
        self,
        scoring_system=None,
        include_weekly: bool = True,
        include_monthly: bool = True
    ):
        """
        初始化多周期分析器

        Args:
            scoring_system: 评分系统实例
            include_weekly: 是否包含周线分析
            include_monthly: 是否包含月线分析
        """
        self.scoring_system = scoring_system
        self.include_weekly = include_weekly
        self.include_monthly = include_monthly

    def analyze_timeframe_alignment(
        self,
        daily_data: pd.DataFrame,
        weekly_data: Optional[pd.DataFrame] = None,
        monthly_data: Optional[pd.DataFrame] = None
    ) -> MultiTimeframeResult:
        """
        检查跨周期信号一致性

        Args:
            daily_data: 日线数据
            weekly_data: 周线数据（可选）
            monthly_data: 月线数据（可选）

        Returns:
            MultiTimeframeResult: 多周期分析结果
        """
        # 计算各周期评分
        daily_score = self._calculate_timeframe_score(daily_data, 'daily')

        weekly_score = None
        if self.include_weekly and weekly_data is not None and len(weekly_data) > 20:
            weekly_score = self._calculate_timeframe_score(weekly_data, 'weekly')

        monthly_score = None
        if self.include_monthly and monthly_data is not None and len(monthly_data) > 10:
            monthly_score = self._calculate_timeframe_score(monthly_data, 'monthly')

        # 确定对齐状态
        alignment = self._determine_alignment(daily_score, weekly_score, monthly_score)

        # 计算综合评分
        combined_score, alignment_bonus = self._calculate_combined_score(
            daily_score, weekly_score, monthly_score, alignment
        )

        # 计算置信度
        confidence = self._calculate_confidence(
            daily_score, weekly_score, monthly_score, alignment
        )

        return MultiTimeframeResult(
            daily=daily_score,
            weekly=weekly_score,
            monthly=monthly_score,
            alignment=alignment,
            combined_score=combined_score,
            alignment_bonus=alignment_bonus,
            confidence=confidence
        )

    def _calculate_timeframe_score(
        self,
        df: pd.DataFrame,
        timeframe: str
    ) -> TimeframeScore:
        """计算单周期评分"""
        if df is None or len(df) < 10:
            return TimeframeScore(
                timeframe=timeframe,
                score=50.0,
                trend='sideway',
                strength=0.0,
                signal='hold'
            )

        # 如果有评分系统，使用评分系统
        if self.scoring_system is not None:
            try:
                result = self.scoring_system.calculate_comprehensive_score(df)
                score = result.get('final_score', 50.0)
            except Exception:
                score = self._calculate_simple_score(df)
        else:
            score = self._calculate_simple_score(df)

        # 判断趋势
        trend, strength = self._determine_trend(df)

        # 判断信号
        signal = self._determine_signal(score, trend)

        return TimeframeScore(
            timeframe=timeframe,
            score=score,
            trend=trend,
            strength=strength,
            signal=signal
        )

    def _calculate_simple_score(self, df: pd.DataFrame) -> float:
        """简单评分计算（无评分系统时使用）"""
        if len(df) < 20:
            return 50.0

        # 基于均线和价格位置
        close = df['close'].iloc[-1]
        ma5 = df['close'].rolling(5).mean().iloc[-1]
        ma10 = df['close'].rolling(10).mean().iloc[-1]
        ma20 = df['close'].rolling(20).mean().iloc[-1]

        # 均线得分
        score = 50.0
        if close > ma5 > ma10 > ma20:
            score = 70.0
        elif close > ma5 and close > ma10:
            score = 60.0
        elif close < ma5 < ma10 < ma20:
            score = 30.0
        elif close < ma5 and close < ma10:
            score = 40.0

        return score

    def _determine_trend(self, df: pd.DataFrame) -> Tuple[str, float]:
        """判断趋势方向和强度"""
        if len(df) < 20:
            return 'sideway', 0.0

        # 使用均线斜率判断趋势
        ma5 = df['close'].rolling(5).mean()
        ma20 = df['close'].rolling(20).mean()

        # 计算斜率
        if len(ma5) >= 5:
            ma5_slope = (ma5.iloc[-1] - ma5.iloc[-5]) / ma5.iloc[-5]
        else:
            ma5_slope = 0

        if len(ma20) >= 10:
            ma20_slope = (ma20.iloc[-1] - ma20.iloc[-10]) / ma20.iloc[-10]
        else:
            ma20_slope = 0

        # 综合判断
        avg_slope = (ma5_slope + ma20_slope) / 2

        if avg_slope > 0.02:
            return 'up', min(1.0, avg_slope * 10)
        elif avg_slope < -0.02:
            return 'down', min(1.0, abs(avg_slope) * 10)
        else:
            return 'sideway', 0.5

    def _determine_signal(self, score: float, trend: str) -> str:
        """判断信号"""
        if score >= 70 and trend == 'up':
            return 'buy'
        elif score <= 30 and trend == 'down':
            return 'sell'
        else:
            return 'hold'

    def _determine_alignment(
        self,
        daily: TimeframeScore,
        weekly: Optional[TimeframeScore],
        monthly: Optional[TimeframeScore]
    ) -> TimeframeAlignment:
        """确定周期对齐状态"""
        # 检查数据可用性
        if weekly is None and monthly is None:
            return TimeframeAlignment.INSUFFICIENT_DATA

        signals = [daily.signal]
        trends = [daily.trend]

        if weekly is not None:
            signals.append(weekly.signal)
            trends.append(weekly.trend)

        if monthly is not None:
            signals.append(monthly.signal)
            trends.append(monthly.trend)

        # 检查信号一致性
        unique_signals = set(s for s in signals if s != 'hold')
        unique_trends = set(t for t in trends if t != 'sideway')

        if len(unique_signals) <= 1 and len(unique_trends) <= 1:
            if len(signals) >= 3:
                return TimeframeAlignment.FULLY_ALIGNED
            else:
                return TimeframeAlignment.PARTIALLY_ALIGNED
        elif len(unique_signals) > 1 or len(unique_trends) > 1:
            return TimeframeAlignment.DIVERGENT
        else:
            return TimeframeAlignment.PARTIALLY_ALIGNED

    def _calculate_combined_score(
        self,
        daily: TimeframeScore,
        weekly: Optional[TimeframeScore],
        monthly: Optional[TimeframeScore],
        alignment: TimeframeAlignment
    ) -> Tuple[float, float]:
        """计算综合评分"""
        # 加权平均
        weighted_score = daily.score * self.TIMEFRAME_WEIGHTS['daily']

        total_weight = self.TIMEFRAME_WEIGHTS['daily']

        if weekly is not None:
            weighted_score += weekly.score * self.TIMEFRAME_WEIGHTS['weekly']
            total_weight += self.TIMEFRAME_WEIGHTS['weekly']

        if monthly is not None:
            weighted_score += monthly.score * self.TIMEFRAME_WEIGHTS['monthly']
            total_weight += self.TIMEFRAME_WEIGHTS['monthly']

        base_score = weighted_score / total_weight if total_weight > 0 else 50.0

        # 对齐奖励
        alignment_bonus = self.ALIGNMENT_BONUS.get(alignment, 0.0)

        # 最终评分 = 基础评分 × (1 + 对齐奖励)
        combined_score = base_score * (1 + alignment_bonus)

        # 限制在0-100范围
        combined_score = max(0, min(100, combined_score))

        return combined_score, alignment_bonus

    def _calculate_confidence(
        self,
        daily: TimeframeScore,
        weekly: Optional[TimeframeScore],
        monthly: Optional[TimeframeScore],
        alignment: TimeframeAlignment
    ) -> float:
        """计算置信度"""
        base_confidence = 0.5  # 基础置信度

        # 数据完整性加分
        if weekly is not None:
            base_confidence += 0.15
        if monthly is not None:
            base_confidence += 0.10

        # 对齐状态加分
        alignment_scores = {
            TimeframeAlignment.FULLY_ALIGNED: 0.25,
            TimeframeAlignment.PARTIALLY_ALIGNED: 0.15,
            TimeframeAlignment.DIVERGENT: -0.10,
            TimeframeAlignment.INSUFFICIENT_DATA: 0.0
        }
        base_confidence += alignment_scores.get(alignment, 0.0)

        # 信号强度加分
        if daily.signal != 'hold':
            base_confidence += 0.05

        return max(0.0, min(1.0, base_confidence))

    def resample_to_weekly(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        将日线数据重采样为周线

        Args:
            daily_data: 日线数据

        Returns:
            DataFrame: 周线数据
        """
        if 'timestamp' not in daily_data.columns:
            daily_data = daily_data.copy()
            daily_data['timestamp'] = daily_data.index

        daily_data = daily_data.set_index('timestamp')

        weekly = daily_data.resample('W').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return weekly.reset_index()

    def resample_to_monthly(self, daily_data: pd.DataFrame) -> pd.DataFrame:
        """
        将日线数据重采样为月线

        Args:
            daily_data: 日线数据

        Returns:
            DataFrame: 月线数据
        """
        if 'timestamp' not in daily_data.columns:
            daily_data = daily_data.copy()
            daily_data['timestamp'] = daily_data.index

        daily_data = daily_data.set_index('timestamp')

        monthly = daily_data.resample('M').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()

        return monthly.reset_index()


def analyze_multi_timeframe(
    daily_data: pd.DataFrame,
    scoring_system=None
) -> MultiTimeframeResult:
    """
    便捷函数：执行多周期分析

    Args:
        daily_data: 日线数据
        scoring_system: 评分系统实例

    Returns:
        MultiTimeframeResult: 分析结果
    """
    analyzer = MultiTimeframeAnalyzer(scoring_system=scoring_system)

    # 重采样到周线和月线
    weekly_data = analyzer.resample_to_weekly(daily_data)
    monthly_data = analyzer.resample_to_monthly(daily_data)

    return analyzer.analyze_timeframe_alignment(
        daily_data, weekly_data, monthly_data
    )