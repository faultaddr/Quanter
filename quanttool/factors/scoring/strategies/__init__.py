"""Scoring strategies for QuantTool."""

from quanttool.factors.scoring.strategies.trend import TrendScoringStrategy
from quanttool.factors.scoring.strategies.breakout import BreakoutScoringStrategy
from quanttool.factors.scoring.strategies.multi_dimension import MultiDimensionScoringStrategy

__all__ = [
    'TrendScoringStrategy',
    'BreakoutScoringStrategy',
    'MultiDimensionScoringStrategy',
]
