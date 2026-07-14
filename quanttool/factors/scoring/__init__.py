"""Unified scoring system for QuantTool.

This module provides a strategy-based scoring framework that supports
multiple scoring strategies with a unified interface.

Architecture:
- ScoringStrategy: Base class for all scoring strategies
- ScoreResult: Data class for scoring results
- UnifiedScoringSystem: Facade that combines multiple strategies

Available Strategies:
- TrendScoringStrategy: Trend-following score
- BreakoutScoringStrategy: Low-position breakout score
- MultiDimensionScoringStrategy: Multi-factor score (MyTT based)
"""

from quanttool.factors.scoring.base import (
    ScoreResult,
    ScoringStrategy,
)
from quanttool.factors.scoring.unified_scoring_system import UnifiedScoringSystem

# Import strategies for convenience
from quanttool.factors.scoring.strategies import (
    TrendScoringStrategy,
    BreakoutScoringStrategy,
    MultiDimensionScoringStrategy,
)

__all__ = [
    # Base classes
    'ScoreResult',
    'ScoringStrategy',
    # Main facade
    'UnifiedScoringSystem',
    # Strategies
    'TrendScoringStrategy',
    'BreakoutScoringStrategy',
    'MultiDimensionScoringStrategy',
]
