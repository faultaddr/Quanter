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
        result = self._legacy_system.calculate_all_scores(
            df=df,
            stock_code=kwargs.get("symbol", ""),
            trade_date_T=kwargs.get("trade_date", ""),
            trade_date_T1=kwargs.get("trade_date_t1"),
            open_T1=kwargs.get("open_t1"),
        )

        if "error" in result:
            return ScoreResult(
                final_score=0,
                passed_filter=False,
                filter_reason=result["error"],
                strategy_name=self.name,
                details={"legacy_result": result},
            )

        latest = df.iloc[-1]
        factors_raw = result.get("factors_raw", {})
        position_modifier, pos_warnings, protection_info = (
            self._legacy_system._calculate_position_modifier(latest, factors_raw)
        )
        all_warnings = result.get("warnings", []) + pos_warnings

        return ScoreResult(
            final_score=result.get("score", result.get("final_score", 0)),
            passed_filter=result.get("bias_passed", True),
            filter_reason=result.get("filter_reason", ""),
            strategy_name=self.name,
            details={
                "legacy_result": result,
                "trend_score": result.get("trend_score", 0),
                "momentum_score": result.get("momentum_score", 0),
                "money_score": result.get("money_score", 0),
                "trend_bonus": result.get("trend_bonus", 0),
                "volume_bonus": result.get("volume_bonus", 0),
                "trigger_type": result.get("trigger_type", "none"),
                "trigger_detail": result.get("trigger_detail", ""),
                "position_modifier": position_modifier,
                "position_protection": protection_info,
                "factors_raw": factors_raw,
                "factors_score": result.get("factors_score", {}),
                "execution": result.get("execution", {}),
                "warnings": all_warnings,
                "score_grade": result.get("score_grade", "一般"),
            },
        )
