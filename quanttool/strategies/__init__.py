"""Strategies package for QuantTool."""

# Import strategies to register them
from . import ma_cross, breakout
from . import rsi, bollinger, macd, kdj, dual_ma, turtle, ma_alignment

__all__ = [
    'ma_cross',
    'breakout',
    'rsi',
    'bollinger',
    'macd',
    'kdj',
    'dual_ma',
    'turtle',
    'ma_alignment'
]