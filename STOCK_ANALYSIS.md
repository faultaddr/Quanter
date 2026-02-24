# Stock Analysis with Technical Indicators

QuantTool now includes advanced stock analysis capabilities with comprehensive technical indicators and trading strategy evaluation.

## Features

- **Over 30 Technical Indicators**: Including RSI, MACD, KDJ, Bollinger Bands, MA, CCI, DMI, TRIX, VR, CR, and more
- **Multiple Trading Strategies**: RSI, MACD, Moving Average Crossover, Bollinger Bands, and Combined strategies
- **Real-time Signal Evaluation**: Determines current buy/sell signals based on technical analysis
- **Detailed Reports**: Comprehensive analysis with current market data, indicators, and recommendations

## Installation

```bash
pip install -e .
```

## Usage

### Command Line Interface

Analyze a specific stock:

```bash
quant analyze analyze <stock_code> [options]
```

Examples:

```bash
# Analyze 601777 for the past 360 days (default)
quant analyze analyze 601777

# Analyze 601777 for the past 30 days
quant analyze analyze 601777 --days 30

# Save analysis to file
quant analyze analyze 601777 --days 90 --output analysis_report.txt
```

### Available Options

- `--days, -d`: Number of days to analyze (default: 360)
- `--output, -o`: Output file to save the analysis report

### Implemented Indicators

#### Momentum Indicators
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- KDJ (Stochastic Oscillator)
- CCI (Commodity Channel Index)
- WR (Williams %R)

#### Trend Indicators
- Moving Averages (MA5, MA10, MA20, MA30, MA50, MA100, MA200)
- Bollinger Bands
- BBI (Bull and Bear Index)
- DMI (Directional Movement Index)

#### Volume Indicators
- OBV (On Balance Volume)
- VR (Volume Ratio)
- EMV (Ease of Movement)

#### Volatility Indicators
- ATR (Average True Range)
- Bollinger Bands Width

## How It Works

1. **Data Retrieval**: Fetches historical stock data from supported data providers (EastMoney, Tushare)
2. **Indicator Calculation**: Calculates over 30 technical indicators using MyTT-based functions
3. **Strategy Evaluation**: Applies multiple trading strategies to generate buy/sell signals
4. **Signal Combination**: Combines signals from different strategies for stronger recommendations
5. **Report Generation**: Creates a comprehensive analysis report with current data and recommendations

## Trading Strategies

### RSI Strategy
- Buy when RSI < 30 (oversold condition)
- Sell when RSI > 70 (overbought condition)

### MACD Strategy
- Buy when MACD line crosses above signal line
- Sell when MACD line crosses below signal line

### Moving Average Crossover
- Buy when short-term MA crosses above long-term MA
- Sell when short-term MA crosses below long-term MA

### Bollinger Bands Strategy
- Buy when price touches lower band
- Sell when price touches upper band

### Combined Strategy
- Evaluates multiple indicators simultaneously
- Assigns strength levels (STRONG_BUY, WEAK_BUY, etc.)

## Output Interpretation

The analysis report includes:

- Current market data (price, volume, daily change)
- Values for all calculated technical indicators
- Signal evaluations from each strategy
- Overall recommendation based on combined signals

## Important Disclaimers

- This analysis is for educational purposes only
- Investment decisions should be made based on comprehensive research and personal judgment
- Past performance does not guarantee future results
- Trading involves substantial risk