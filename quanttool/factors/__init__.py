"""Factors package for QuantTool."""

# Import factors to register them
from .technical import momentum, volatility

__all__ = ['momentum', 'volatility']