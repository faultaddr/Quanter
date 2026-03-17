# QuantTool - A-Share Quantitative Trading Analysis Platform

QuantTool is a quantitative analysis platform designed specifically for the A-share market, featuring technical analysis, pattern recognition, multi-factor scoring, and intelligent circuit breakers.

## Core Features

### 1. Multi-Dimensional Technical Analysis Scoring System

A layered scoring architecture based on three major factor categories:

```
Final Score = Trend Score × Position Modifier
```

#### Trend Factors (40% Weight)
| Factor | Weight | Description |
|--------|--------|-------------|
| Trend Strength | 20% | MA20 deviation, DMI state correction |
| MA Slope | 20% | MA5 slope for trend direction |
| MACD Momentum | 20% | MACD histogram changes |
| Money Flow | 20% | OBV money flow score |
| Volume | 10% | Price-volume coordination |
| Candlestick Pattern | 10% | Position-sensitive scoring |

### 2. Candlestick Pattern Recognition System

Supports 20+ classic candlestick patterns:

#### Single Candle Patterns
- **Bullish**: Hammer, Inverted Hammer, Big White Candle
- **Bearish**: Hanging Man, Shooting Star, Big Black Candle
- **Neutral**: Doji, Long Upper Shadow, Long Lower Shadow, Spinning Top

#### Multi-Candle Patterns
- **Bullish Combinations**: Bullish Engulfing, Morning Star, Piercing Line
- **Bearish Combinations**: Bearish Engulfing, Evening Star, Dark Cloud Cover

#### Pattern Visualization
- ASCII art candlestick charts (color-coded: 🟢 green bullish / 🔴 red bearish)
- Pattern illustration diagrams

### 3. Intelligent Circuit Breaker Mechanism

Multi-layer risk control system:

| Circuit Breaker Rule | Trigger Condition | Effect |
|---------------------|-------------------|--------|
| Strong Bearish Breaker | Big black candle + high position | Score drops to ≤25, position 0% |
| High Position Bearish Breaker | High position + bearish pattern | Score drops to ≤25, position 0% |
| Bull Trap Breaker | High position + bullish + extreme overbought | Force hold |
| Extreme Overbought Breaker | 3+ indicators maxed out | Position modifier ≤0.30 |

### 4. Built-in Trading Strategies

| Strategy Name | Type | Description |
|---------------|------|-------------|
| `ma_cross` | Trend Following | Moving average crossover (golden cross buy, death cross sell) |
| `dual_ma` | Trend Following | Dual MA strategy (customizable periods) |
| `breakout` | Breakout | Price breaks N-day high/low |
| `turtle` | Trend Following | Turtle trading strategy (Donchian Channel) |
| `ma_alignment` | Trend Following | MA bullish alignment strategy |
| `rsi` | Oscillator | RSI overbought/oversold strategy |
| `macd` | Trend Indicator | MACD golden/death cross strategy |
| `kdj` | Oscillator | KDJ golden/death cross strategy |
| `bollinger` | Oscillator | Bollinger Band mean reversion strategy |

### 5. Four-Section Report Architecture

```
Part 1: Core Conclusion
├── Action Signal (Buy/Sell/Hold)
├── Technical Score (0-100)
├── Confidence Level
└── Key Reasons

Part 2: Multi-Dimensional Signal Resonance
├── Trend State (DMI + MA alignment)
├── Momentum State (MACD + RSI)
├── Position State (Bollinger + CCI + WR)
├── Pattern Analysis (Candlestick patterns)
└── Candlestick Chart Visualization

Part 3: Quantitative Score Breakdown
├── Final score formula
├── Factor score details
├── Position modifier
└── Red/Black list

Part 4: Trading Execution Plan
├── Strategy type determination
├── Entry/Stop-loss/Target levels
├── Position sizing
└── Risk warnings
```

## Installation

```bash
git clone https://github.com/yourusername/quanttool.git
cd quanttool
pip install -e .
```

## Quick Start

### 1. Configure Data Source

```bash
export TUSHARE_TOKEN="your_tushare_token"
```

### 2. Analyze Stocks

```bash
# Analyze a single stock
quant analyze 000001.SZ

# Specify period
quant analyze 000001.SZ --days 360

# Market scan
quant analyze scan --market all --days 360 --top 10
```

### 3. Backtest Strategies

```bash
# MA Cross strategy
quant backtest run --strategy ma_cross --symbol 000001.SZ \
  --start 2023-01-01 --end 2023-06-01 --cash 100000

# RSI strategy
quant backtest run --strategy rsi --symbol 000001.SZ \
  --start 2023-01-01 --end 2023-06-01

# Turtle strategy
quant backtest run --strategy turtle --symbol 000001.SZ \
  --start 2023-01-01 --end 2023-06-01
```

## Project Architecture

```
quanttool/
├── factors/                    # Factor analysis module
│   ├── scoring_system.py       # Multi-dimensional scoring system
│   ├── talib_patterns.py       # Candlestick pattern recognition (TA-Lib 61 patterns)
│   ├── stock_analyzer.py       # Stock comprehensive analysis
│   └── tech_indicators.py      # Technical indicator calculation
├── strategies/                 # Trading strategies module
│   ├── ma_cross.py            # MA crossover strategy
│   ├── dual_ma.py             # Dual MA strategy
│   ├── breakout.py            # Breakout strategy
│   ├── turtle.py              # Turtle strategy
│   ├── ma_alignment.py        # MA alignment strategy
│   ├── rsi.py                 # RSI strategy
│   ├── macd.py                # MACD strategy
│   ├── kdj.py                 # KDJ strategy
│   └── bollinger.py           # Bollinger Band strategy
├── infrastructure/             # Infrastructure layer
│   ├── data_providers/        # Data providers
│   └── stores/                # Storage layer
├── reports/                    # Report generation
├── cli/                        # Command-line interface
└── web/                        # Web API
```

## Python API

```python
from quanttool.factors.stock_analyzer import StockAnalyzer
from quanttool.factors.talib_patterns import (
    TalibPatternRecognizer,
    draw_candlestick_chart,
    draw_pattern_illustration
)

# Analyze stock
analyzer = StockAnalyzer()
report = analyzer.analyze_stock("000001.SZ", days=360)
print(report)

# Recognize candlestick patterns (TA-Lib 61 patterns)
recognizer = TalibPatternRecognizer()
patterns = recognizer.recognize_all(df, lookback=5)

# Draw candlestick chart
chart = draw_candlestick_chart(df, num_candles=10)
print(chart)

# Get pattern illustration
illustration = draw_pattern_illustration("晨星")
print(illustration)
```

## Report Example

```markdown
## Part 1: Core Conclusion

### 🟢 Action Signal: Buy

**Technical Score: 70.5 (Good)**

### 📊 Recent Candlestick Chart

High: ¥63.77
   │        │    │  ████ │     │
   │        │    │  ████ │     │
   │   │    ████ │  ████ ▓▓▓▓  │
   │   │    ████ │  ████ ▓▓▓▓  │
Low: ¥60.27

> Legend: 🟢 Green = Bullish candle | 🔴 Red = Bearish candle
```

## License

MIT License