# TA-Lib Installation Guide

## Overview

The quantitative trading system uses technical analysis indicators provided by TA-Lib. If the native TA-Lib library cannot be installed, the system will automatically fall back to a pure Python implementation that maintains the same interface.

## Installation Options

### Option 1: Full Installation (Recommended)

Run the setup script to install both the C library and Python bindings:

```bash
./setup_talib.sh
```

This script will:
1. Install system dependencies (build tools)
2. Download, compile and install the TA-Lib C library
3. Install the Python TA-Lib package
4. Verify the installation

### Option 2: Skip Native Installation

If you encounter issues with the native installation, the system will work with the pure Python mock implementation which covers all the technical indicators used in the strategies.

## Technical Details

The system intelligently handles both scenarios:

- If TA-Lib is installed: Uses the native library for optimal performance
- If TA-Lib is not available: Falls back to pure Python implementation with similar functionality

## Supported Indicators

The mock implementation supports all indicators used by the strategies:

- SMA - Simple Moving Average
- EMA - Exponential Moving Average
- RSI - Relative Strength Index
- MACD - Moving Average Convergence Divergence
- BBANDS - Bollinger Bands
- STOCH - Stochastic Oscillator
- ATR - Average True Range
- ADX - Average Directional Index
- WILLR - Williams' %R
- ROC - Rate of Change
- CCI - Commodity Channel Index
- CDLHAMMER - Hammer candlestick pattern
- CDLENGULFING - Engulfing candlestick pattern
- MOM - Momentum indicator
- STDDEV - Standard Deviation
- VAR - Variance

## Verification

To verify the installation works, run:

```python
from quant_trade_a_share.strategies.advanced_strategies import AdvancedStrategyManager
manager = AdvancedStrategyManager()
print(f"Strategies available: {len(manager.get_strategy_names())}")
```

If successful, you should see a list of available strategies and the system will indicate whether TA-Lib is available or if it's using the mock implementation.