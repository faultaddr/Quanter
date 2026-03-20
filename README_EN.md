# QuantTool - A-Share Quantitative Trading Analysis Platform

[中文](./README.md) | [繁體中文](./README_ZH_TW.md)

QuantTool is a professional A-share quantitative trading analysis platform, providing technical analysis, factor research, strategy backtesting, and risk control capabilities.

## Key Features

- **Free Data Sources**: Prioritizes free data sources like Ashare, EastMoney, AkShare
- **Real-time Data**: Supports minute-level real-time market data
- **Web Interface**: Modern web frontend with full functionality
- **Strategy Backtesting**: Multiple technical indicator strategies with realistic A-share trading simulation
- **Factor Research**: IC/IR analysis, factor optimization, factor neutralization
- **Risk Management**: Industry exposure monitoring, blacklist checking, position shrinking

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

Supports recognition of 20+ classic candlestick patterns including hammer lines, engulfing patterns, morning/evening stars.

### 3. Factor Research & Optimization

| Feature | Description |
|---------|-------------|
| Factor Validation | IC/IR analysis, factor predictive ability assessment |
| Factor Optimization | IR-weighted, IC-weighted, equal weight, risk parity |
| Factor Neutralization | Industry neutral, market cap neutral |
| Factor Pipeline | Winsorize, Standardize processing |

### 4. Portfolio Risk Management

| Feature | Description |
|---------|-------------|
| Industry Exposure | Single industry position limit (default 20%) |
| Blacklist Check | Prohibited stock monitoring |
| Position Shrinking | Dynamic position adjustment based on drawdown |
| Risk Scoring | Multi-dimensional risk assessment |

### 5. A-Share Trading Constraints

| Constraint Type | Description |
|-----------------|-------------|
| Limit Up/Down | Cannot buy at limit up, cannot sell at limit down |
| ST Stock Restriction | Configurable ST stock exclusion |
| Real Trading Costs | Commission, stamp duty, slippage simulation |

### 6. Built-in Trading Strategies

| Strategy Name | Type | Description |
|---------------|------|-------------|
| `ma_cross` | Trend Following | Moving average crossover |
| `dual_ma` | Trend Following | Dual MA strategy |
| `breakout` | Breakout | Price breaks N-day high/low |
| `turtle` | Trend Following | Turtle trading strategy |
| `ma_alignment` | Trend Following | MA bullish alignment |
| `rsi` | Oscillator | RSI overbought/oversold |
| `macd` | Trend Indicator | MACD golden/death cross |
| `kdj` | Oscillator | KDJ golden/death cross |
| `bollinger` | Oscillator | Bollinger Band mean reversion |

## Architecture

```
QuantTool/
├── quanttool/
│   ├── core/                    # Core utilities
│   │   ├── errors.py           # Error handling
│   │   ├── logging.py          # Logging
│   │   └── registry.py         # Component registry
│   │
│   ├── domain/                  # Domain layer
│   │   ├── interfaces/         # Interface definitions
│   │   └── models/             # Data models
│   │
│   ├── application/             # Application services
│   │   ├── analysis_service.py
│   │   ├── backtest_service.py
│   │   └── factor_service.py
│   │
│   ├── infrastructure/          # Infrastructure layer
│   │   ├── data_providers/     # Data providers
│   │   │   ├── ashare_provider.py
│   │   │   ├── akshare_minute_provider.py
│   │   │   └── data_fetcher.py
│   │   └── stores/             # Storage layer
│   │
│   ├── strategies/              # Trading strategies
│   │   ├── ma_cross.py
│   │   ├── breakout.py
│   │   └── ...
│   │
│   ├── factors/                 # Factor library
│   │   ├── factor_validator.py
│   │   ├── factor_pipeline.py
│   │   ├── factor_registry.py
│   │   └── neutralizer.py
│   │
│   ├── optimization/            # Optimizers
│   │   └── weight_optimizer.py
│   │
│   ├── risk/                    # Risk management
│   │   └── risk_controller.py
│   │
│   ├── backtest/                # Backtest engine
│   │   ├── engine.py
│   │   └── ashare_constraints.py
│   │
│   ├── web/                     # Web layer
│   │   ├── api/                # API routes
│   │   └── frontend/           # Next.js frontend
│   │
│   └── cli/                     # CLI tools
│       └── main.py
│
└── tests/                       # Test cases
```

## Installation

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm or yarn

### Setup

```bash
# Clone project
git clone https://github.com/faultaddr/Quanter.git
cd Quanter

# Install Python dependencies
pip install -e .

# Install frontend dependencies
cd quanttool/web/frontend
npm install
```

## Quick Start

### Start Services

```bash
# Start backend service
uvicorn quanttool.web.app:app --host 0.0.0.0 --port 8000

# Start frontend dev server
cd quanttool/web/frontend
npm run dev
```

Visit http://localhost:3000 to open the web interface.

### Web Interface Pages

| Page | Function |
|------|----------|
| `/` | Market overview, market indices |
| `/analyze` | Stock analysis, K-line, technical indicators |
| `/backtest` | Strategy backtest, returns comparison |
| `/factors` | Factor research, IC/IR analysis |
| `/risk` | Portfolio risk management |
| `/scan` | Smart stock screening |
| `/picks` | AI recommended stocks |
| `/monitor` | Real-time market monitoring |
| `/model` | ML model training & prediction |

### Using CLI

```bash
# Analyze stock
quant analyze 600519 --days 360

# Backtest strategy
quant backtest run --strategy ma_cross --symbol 600519 \
  --start 2023-01-01 --end 2024-01-01 --cash 100000
```

### Using Python API

```python
from quanttool.factors.stock_analyzer import StockAnalyzer
from quanttool.backtest.engine import BacktestEngine

# Analyze stock
analyzer = StockAnalyzer()
report = analyzer.analyze_stock("600519", days=360)
print(report.summary)

# Backtest strategy
engine = BacktestEngine()
result = engine.run(
    symbol="600519",
    strategy="ma_cross",
    start_date="2023-01-01",
    end_date="2024-01-01",
    initial_capital=1000000,
)
print(f"Return: {result.total_return:.2%}")
```

## Data Source Priority

1. **Ashare** - Free, no Token required, primary source
2. **EastMoney** - Free, rich data
3. **AkShare** - Free, comprehensive APIs
4. **TuShare** - Requires Token, fallback

## Performance

| Operation | P50 | P95 |
|-----------|-----|-----|
| Cache Hit | < 10ms | < 50ms |
| Data Fetch | < 500ms | < 2s |
| Full Analysis | < 2s | < 5s |

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run coverage
pytest tests/ --cov=quanttool --cov-report=html
```

Current test coverage: 400+ test cases passing.

## License

MIT License
