# Quantitative Trading Tool for A-Share Market

## Overview

This is a comprehensive quantitative trading tool designed specifically for the Chinese A-share market. It provides backtesting capabilities using free data sources and includes multiple trading strategies commonly used in quantitative finance.

## Features

- **Multi-source Data Fetching**: Access to various free data sources including EastMoney, Tushare, Baostock, and Yahoo Finance
- **Flexible Backtesting Engine**: Robust backtesting framework with realistic trading simulation
- **Multiple Trading Strategies**: Includes popular strategies like Moving Average Crossover, RSI, Mean Reversion, Bollinger Bands, and MACD
- **Performance Analysis**: Comprehensive performance metrics and visualization tools
- **User-Friendly Interface**: Interactive web-based dashboard for easy strategy testing

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/quant-trade-a-share.git
   cd quant-trade-a-share
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. For Tushare data access (optional), get a free token at https://tushare.pro/ and set it:
   ```python
   from quant_trade_a_share.data.data_fetcher import DataFetcher
   fetcher = DataFetcher()
   fetcher.set_tushare_token('your_token_here')
   ```

## Usage

### Quick Start

```python
from quant_trade_a_share import get_data, run_backtest, analyze_performance
from quant_trade_a_share.strategies import MovingAverageCrossoverStrategy

# Fetch data
data = get_data('000001', '2020-01-01', '2021-01-01', source='eastmoney')

# Define strategy
strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)

# Run backtest
results = run_backtest(strategy, '2020-01-01', '2021-01-01', initial_capital=100000)

# Analyze performance
metrics = analyze_performance(results)
print(metrics)
```

### Using Different Strategies

```python
from quant_trade_a_share.strategies import RSIStrategy, MeanReversionStrategy, BollingerBandsStrategy, MACDStrategy

# RSI Strategy
rsi_strategy = RSIStrategy(rsi_period=14, oversold=30, overbought=70)

# Mean Reversion Strategy
mr_strategy = MeanReversionStrategy(window=20, threshold=1.5)

# Bollinger Bands Strategy
bb_strategy = BollingerBandsStrategy(window=20, num_std=2)

# MACD Strategy
macd_strategy = MACDStrategy(fast_period=12, slow_period=26, signal_period=9)
```

### Running the Web Interface

```bash
cd ui
python app.py
```

Then navigate to http://127.0.0.1:8050/ in your browser.

## Data Sources

### EastMoney (Recommended for A-shares)
- Pros: Comprehensive Chinese market data, actively maintained
- Cons: May have rate limits on free tier
- Usage: Default option in the tool

### Tushare
- Pros: High quality data, extensive coverage
- Cons: Requires registration and token, rate limits on free tier
- Usage: Register at https://tushare.pro/ and set token

### Baostock
- Pros: Good for Chinese stock data, free
- Cons: May have limited historical depth
- Usage: Integrated seamlessly in the tool

### Yahoo Finance
- Pros: Global coverage, well-documented
- Cons: Less reliable for Chinese market data
- Usage: Available as fallback option

## Strategy Details

### Moving Average Crossover
This strategy generates buy signals when a short-term moving average crosses above a long-term moving average, and sell signals when the opposite occurs.

Parameters:
- `short_window`: Short-term moving average window
- `long_window`: Long-term moving average window

### RSI (Relative Strength Index)
This momentum oscillator measures the speed and change of price movements. The strategy buys when RSI moves above an oversold level and sells when it moves below an overbought level.

Parameters:
- `rsi_period`: Look-back period for RSI calculation
- `oversold`: Oversold threshold (typically 30)
- `overbought`: Overbought threshold (typically 70)

### Mean Reversion
This strategy assumes that prices tend to revert to their mean over time. It buys when the price is significantly below the moving average and sells when significantly above.

Parameters:
- `window`: Look-back period for mean calculation
- `threshold`: Number of standard deviations to trigger signal

### Bollinger Bands
This strategy uses bands placed above and below a moving average. It buys when the price touches the lower band and sells when touching the upper band.

Parameters:
- `window`: Look-back period for moving average
- `num_std`: Number of standard deviations for band width

### MACD (Moving Average Convergence Divergence)
This trend-following momentum indicator shows the relationship between two moving averages of a security's price. The strategy buys when the MACD line crosses above the signal line and sells when below.

Parameters:
- `fast_period`: Fast EMA period
- `slow_period`: Slow EMA period
- `signal_period`: Signal line EMA period

## Performance Metrics

The tool calculates several important performance metrics:

- **Total Return**: Overall percentage return of the strategy
- **Annualized Return**: Average yearly return
- **Volatility**: Annualized standard deviation of returns
- **Sharpe Ratio**: Risk-adjusted return (excess return per unit of risk)
- **Max Drawdown**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit divided by gross loss
- **Value at Risk (VaR)**: Expected maximum loss at a given confidence level
- **Sortino Ratio**: Similar to Sharpe ratio but only considers downside risk

## Risk Management

The backtesting engine incorporates several risk management features:

- Transaction costs: Configurable commission rates applied to each trade
- Capital constraints: Ensures sufficient funds before executing trades
- Position sizing: Calculates maximum position size based on available capital

## Limitations

- **Past Performance**: Results are based on historical data and don't guarantee future performance
- **Data Quality**: Relies on free data sources which may have inaccuracies
- **Market Conditions**: Strategies may perform differently under varying market conditions
- **Transaction Costs**: Simplified commission model may not reflect real-world costs accurately

## Contributing

Contributions to improve the tool are welcome. Please submit pull requests with enhancements or bug fixes.

## License

This project is licensed under the MIT License - see the LICENSE file for details.