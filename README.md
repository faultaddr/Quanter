# Quantitative Trading Tool for A-Share Market

A comprehensive quantitative trading tool designed specifically for the Chinese A-share market. The tool provides data fetching, strategy backtesting, and performance analysis capabilities.

## Features

- **Data Fetching**: Retrieve historical stock data from Chinese exchanges (Shanghai and Shenzhen)
- **Strategy Backtesting**: Test trading strategies against historical data
- **Performance Analysis**: Comprehensive performance metrics and risk analysis
- **Multiple Strategies**: Support for various trading strategies (MA crossover, RSI, mean reversion, etc.)
- **Web Interface**: Interactive dashboard for strategy testing and visualization

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd quant_trade_a_share
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

### Basic Usage

```python
from quant_trade_a_share import run_backtest, analyze_performance
from quant_trade_a_share.strategies import MovingAverageCrossoverStrategy

# Create a strategy
strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)

# Run a backtest
results = run_backtest(
    strategy, 
    '2022-01-01', 
    '2022-12-31', 
    initial_capital=100000,
    symbol='000001'
)

# Analyze performance
metrics = analyze_performance(results)
print(f"Total Return: {metrics['total_return']:.2%}")
print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
```

### Using Mock Data for Testing

For testing purposes when external data sources are unavailable:

```python
from quant_trade_a_share.utils.mock_data_generator import generate_mock_data
from quant_trade_a_share.backtest.backtester import Backtester

# Generate mock data
data = generate_mock_data('000001', '2022-01-01', '2022-12-31')

# Run backtest with mock data
backtester = Backtester()
results = backtester.run_with_data(
    strategy,
    data,
    initial_capital=100000,
    symbol='000001'
)
```

### Web Interface

Run the web interface:

```bash
python -m ui.app
```

Then navigate to `http://127.0.0.1:8050` in your browser.

## Components

- `data/data_fetcher.py`: Module for fetching historical stock data
- `strategies/`: Directory containing various trading strategies
- `backtest/backtester.py`: Backtesting engine
- `analysis/performance_analyzer.py`: Performance analysis tools
- `ui/app.py`: Web interface using Dash
- `utils/`: Utility functions and helpers

## Available Strategies

- **Moving Average Crossover**: Buy when short MA crosses above long MA, sell when opposite
- **RSI Strategy**: Buy when RSI is below oversold threshold, sell when above overbought threshold
- **Mean Reversion**: Trade based on deviation from moving average
- **Bollinger Bands**: Trade based on price position relative to Bollinger Bands
- **MACD**: Trade based on MACD line and signal line crossovers

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License.