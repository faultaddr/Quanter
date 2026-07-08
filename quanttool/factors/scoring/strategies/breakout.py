"""Breakout scoring strategy.

A strategy for finding low-position consolidation breakout stocks.

Core concepts:
- Technical analysis handles "pattern and timing" (low position + consolidation + breakout)
- Factor analysis handles "quality and win rate" (fundamental/valuation/fund/momentum filtering and scoring)

Scoring architecture:
1. Pattern detection layer: Low position + consolidation + breakout detection
2. Factor scoring layer: Quality/growth/valuation/momentum/flow/risk factors
3. Final score = Pattern score × Factor quality score
"""

import pandas as pd
from typing import Dict, Any

from ..base import ScoreResult, ScoringStrategy

# Import the original system for backward compatibility
from ...breakout_scoring_system import BreakoutScoringSystem as _LegacyBreakoutScoringSystem


class BreakoutScoringStrategy(ScoringStrategy):
    """低位盘整突破评分策略

    寻找处于低位、经历盘整后放量突破的股票
    """

    def __init__(
        self,
        min_consolidation_days: int = 20,
        max_consolidation_days: int = 40,
        max_range: float = 0.18,
        min_range: float = 0.12,
        **kwargs
    ):
        """
        初始化突破评分策略

        Args:
            min_consolidation_days: 最少盘整天数
            max_consolidation_days: 最多盘整天数
            max_range: 最大振幅
            min_range: 最小振幅
        """
        self.min_consolidation_days = min_consolidation_days
        self.max_consolidation_days = max_consolidation_days
        self.max_range = max_range
        self.min_range = min_range
        legacy_kwargs = {
            key: kwargs[key]
            for key in ("min_amount", "min_list_days")
            if key in kwargs
        }
        self._legacy_system = _LegacyBreakoutScoringSystem(**legacy_kwargs)
        self._legacy_system.CONSOLIDATION_PARAMS = {
            **self._legacy_system.CONSOLIDATION_PARAMS,
            'min_days': min_consolidation_days,
            'max_days': max_consolidation_days,
            'max_range': max_range,
            'min_range': min_range,
        }

    @property
    def name(self) -> str:
        return "breakout"

    @property
    def description(self) -> str:
        return "低位盘整突破评分策略 - 寻找低位盘整后放量突破的股票"

    def get_required_data_length(self) -> int:
        return 250

    def calculate_score(self, df: pd.DataFrame, **kwargs) -> ScoreResult:
        """计算突破评分"""
        result = self._legacy_system.calculate_score(df)
        details = dict(result.details or {})
        details.update({
            'is_low_position': result.is_low_position,
            'is_consolidating': result.is_consolidating,
            'has_breakout': result.has_breakout,
            'quality_score': result.quality_score,
            'growth_score': result.growth_score,
            'value_score': result.value_score,
            'momentum_score': result.momentum_score,
            'flow_score': result.flow_score,
            'risk_score': result.risk_score,
            'consolidation_days': result.consolidation_days,
            'price_range': result.price_range,
            'volume_ratio': result.volume_ratio,
            'breakout_strength': result.breakout_strength,
        })

        return ScoreResult(
            final_score=result.final_score,
            passed_filter=result.passed_filter,
            filter_reason=result.filter_reason,
            strategy_name=self.name,
            stop_loss_price=result.stop_loss_price,
            take_profit_price=result.take_profit_price,
            details=details,
        )
