"""Strategies package for QuantTool."""

# Import strategies to register them
from . import ma_cross, breakout
from . import rsi, bollinger, macd, kdj, dual_ma, turtle, ma_alignment
from .score_strategy import ScoreStrategy, EnhancedScoreStrategy
from .trend_strategy import TrendStrategy, AdaptiveTrendStrategy
from .trend_momentum_strategy import TrendMomentumStrategy
from .ml_stock_selection_strategy import MLStockSelectionStrategy, MLStockSelector
from .qlib_strategy import QlibStrategy, QlibStockSelector, QlibFeatureEngineer

__all__ = [
    'ma_cross',
    'breakout',
    'rsi',
    'bollinger',
    'macd',
    'kdj',
    'dual_ma',
    'turtle',
    'ma_alignment',
    'ScoreStrategy',
    'EnhancedScoreStrategy',
    'TrendStrategy',
    'AdaptiveTrendStrategy',
    'TrendMomentumStrategy',
    'MLStockSelectionStrategy',
    'MLStockSelector',
    'QlibStrategy',
    'QlibStockSelector',
    'QlibFeatureEngineer',
]