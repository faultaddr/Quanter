"""Multi-dimension scoring strategy.

A comprehensive multi-factor scoring system based on MyTT indicators.

Core design:
1. Scoring formula: Final score = Trend score + Trend confirmation bonus + Volume bonus
2. Three factor groups:
   - Trend factors (confirm direction): MA system, DMI, MACD
   - Momentum factors (confirm strength): MTM, ROC, KDJ, RSI
   - Capital factors (confirm authenticity): OBV, MFI, Volume
3. Right-side trading logic: High score = Trend established + Good right-side trading opportunity
4. First breakout signal: Buy/sell signals only trigger when score first breaks threshold
"""

import pandas as pd
from typing import Dict, Any

from ..base import ScoreResult, ScoringStrategy

# Import the original system for backward compatibility
from ...scoring_system import ScoringSystem as _LegacyScoringSystem


class MultiDimensionScoringStrategy(ScoringStrategy):
    """多维度打分策略（右侧交易版）

    基于 MyTT 指标库设计的多因子组合评分系统
    """

    def __init__(self, **kwargs):
        """
        初始化多维度评分策略

        Args:
            **kwargs: 传递给原始评分系统的参数
        """
        self._legacy_system = _LegacyScoringSystem(**kwargs)

    @property
    def name(self) -> str:
        return "multi_dimension"

    @property
    def description(self) -> str:
        return "多维度评分策略 - 基于MyTT的多因子右侧交易评分系统"

    def get_required_data_length(self) -> int:
        return 200

    def calculate_score(self, df: pd.DataFrame, **kwargs) -> ScoreResult:
        """计算多维度评分"""
        result = self._legacy_system.calculate_score(df)

        # Extract key information from the legacy result
        return ScoreResult(
            final_score=result.get('final_score', 0),
            passed_filter=result.get('passed_filter', True),
            filter_reason=result.get('filter_reason', ''),
            strategy_name=self.name,
            details={
                'trend_score': result.get('trend_score', 0),
                'momentum_score': result.get('momentum_score', 0),
                'capital_score': result.get('capital_score', 0),
                'signal': result.get('signal', 'hold'),
                'signal_strength': result.get('signal_strength', 0),
                'ma_status': result.get('ma_status', {}),
                'macd_status': result.get('macd_status', {}),
                'kdj_status': result.get('kdj_status', {}),
                'rsi_status': result.get('rsi_status', {}),
                'volume_status': result.get('volume_status', {}),
            }
        )
