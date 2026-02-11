# A-Share Quantitative Trading System - User Guide

## Overview
This is a comprehensive quantitative trading system designed specifically for the Chinese A-share market. The system integrates multiple data sources, advanced Qlib-based strategies, and 100+ technical indicators for sophisticated market analysis.

## Key Features

### 1. Multiple Data Sources (Priority Order)
- **Ashare** (Primary) - Direct connection to Chinese stock exchanges
- **EastMoney** - Cookie-based access to premium data
- **Tushare** - Professional financial data API
- **BaoStock** - Free Chinese stock data
- **Yahoo Finance** - International market data

### 2. Advanced Strategy Engine
- **Qlib Integration** - Microsoft's quantitative investment platform
- **100+ Technical Indicators** - Comprehensive factor analysis
- **Multi-Factor Models** - Advanced statistical strategies
- **Machine Learning Models** - AI-powered predictions

### 3. Comprehensive Analysis Tools
- Real-time signal generation
- Risk management
- Backtesting engine
- Performance metrics
- Portfolio optimization

## System Architecture

### Data Layer
- **Ashare Data Fetcher**: Primary data source with direct exchange connectivity
- **EastMoney Integration**: Premium data with cookie authentication
- **Tushare Interface**: Professional API access
- **Fallback Mechanisms**: Automatic switching between data sources

### Strategy Layer
- **Qlib-based Strategies**: Advanced algorithms from Microsoft's platform
- **Traditional Technical Strategies**: Moving averages, RSI, MACD, etc.
- **Custom Factor Models**: 100+ custom indicators
- **Ensemble Methods**: Combined strategy approaches

### Analysis Layer
- **Signal Generation**: Real-time buy/sell signals
- **Risk Management**: Position sizing and stop-losses
- **Backtesting**: Historical strategy validation
- **Performance Analysis**: Risk-return metrics

## Available Strategies

### Qlib-Based Strategies
1. **Alphas101** - Alpha factor models based on Qlib's 101 factors
2. **ML GBDT** - Machine learning gradient boosting decision trees
3. **Technical** - Advanced technical analysis patterns
4. **Ensemble** - Combined model predictions

### Traditional Strategies
1. **MA Crossover** - Moving average crossover signals
2. **RSI** - Relative strength index oversold/overbought
3. **MACD** - Moving average convergence divergence
4. **Bollinger Bands** - Volatility-based trading signals

### Advanced Strategies
1. **Mean Reversion** - Statistical mean reversion models
2. **Momentum** - Trend following strategies
3. **Volume-Based** - Volume-price relationship analysis
4. **Oscillator** - Multiple oscillator combinations
5. **Breakout** - Support/resistance level breaks
6. **Correlation** - Cross-sectional correlation models
7. **Volatility Regime** - Adaptive volatility strategies

## Usage Instructions

### 1. Interactive Mode
```bash
python cli_interface.py --mode interactive
```

Available commands:
- `screen_stocks` - Screen for potentially rising stocks
- `analyze_stock` - Analyze a specific stock
- `predict_stocks` - Predict stock movements
- `run_strategy` - Run a specific strategy
- `gen_signals` - Generate buy/sell signals
- `show_signals` - Show latest signals
- `get_data` - Get stock data
- `calc_indicators` - Calculate technical indicators
- `multi_factor_analysis` - Run 100+ factor analysis
- `run_backtest` - Run strategy backtesting
- `compare_strategies` - Compare multiple strategies

### 2. Direct Analysis
```bash
# Screen stocks
python cli_interface.py --mode screen

# Analyze specific stock
python cli_interface.py --mode analyze --symbol sh600519 --strategy ma_crossover

# Run backtest
python cli_interface.py --mode backtest --symbol 000001.SZ --strategy ma_crossover --start-date 20220101 --end-date 20221231
```

### 3. Multi-Factor Analysis
The system can analyze 100+ technical factors simultaneously:
- Moving averages (5, 10, 20, 30, 50, 60, 120 periods)
- RSI indicators (7, 14, 21, 30 periods)
- MACD combinations (12/26/9, 10/20/5, 5/15/3)
- Bollinger bands (10, 20, 50 periods with 1.5, 2.0, 2.5 std deviations)
- Volume indicators (ratios, z-scores, moving averages)
- Volatility measures (ATR, realized volatility)
- Momentum indicators (ROC, CMF)
- Trend indicators (direction, strength, consistency)

## Configuration

### Data Source Priority
The system automatically prioritizes data sources:
1. First: Ashare (recommended for reliability)
2. Second: EastMoney (premium features)
3. Third: Tushare (professional data)
4. Fourth: BaoStock (free alternative)
5. Fifth: Yahoo Finance (international backup)

### Token Setup
For Tushare access, set your token:
```python
fetcher.set_tushare_token("your_token_here")
```

## Performance Metrics

The system calculates comprehensive performance metrics:
- **Total Return** - Absolute return over the period
- **Annualized Return** - Annual return rate
- **Volatility** - Annualized standard deviation
- **Sharpe Ratio** - Risk-adjusted return
- **Information Ratio** - Excess return per unit of tracking error
- **Maximum Drawdown** - Largest peak-to-trough decline
- **Calmar Ratio** - Annualized return divided by max drawdown

## Troubleshooting

1. **Connection Issues**: The system has fallback mechanisms, but ensure internet connectivity
2. **Data Limitations**: Some data sources may have daily limits; the system automatically switches sources
3. **Token Errors**: Verify Tushare/BaoStock tokens are correctly configured

## System Requirements

- Python 3.7+
- Pandas, NumPy, SciPy
- Qlib 0.9.0+
- Internet connection for data retrieval
- Minimum 8GB RAM recommended for advanced analysis

## Getting Started

1. Install dependencies: `pip install -r requirements.txt`
2. Run the interactive interface: `python cli_interface.py --mode interactive`
3. Use command `help` to see all available commands
4. Start with `screen_stocks` to identify potential opportunities
5. Use `multi_factor_analysis` for comprehensive factor evaluation

The system is production-ready and designed for professional quantitative analysis of the Chinese A-share market.