# Ashare Data Source Integration Report

## Overview
This document summarizes the successful integration of the Ashare data source (https://github.com/mpquant/Ashare) into the A-Share market analysis system. The integration adds support for both Tencent and Sina data sources with fallback mechanisms.

## Integration Details

### 1. Ashare Data Fetcher Module (`quant_trade_a_share/data/ashare_data_fetcher.py`)
- Created a new module implementing the Ashare stock data source
- Supports both Tencent (for daily/minute data) and Sina (for all frequencies) data sources
- Implements fallback mechanisms when primary sources fail
- Provides standardized DataFrame output with OHLCV columns

### 2. DataFetcher Integration (`quant_trade_a_share/data/data_fetcher.py`)
- Added Ashare as a new data source option alongside EastMoney, Tushare, Baostock, and Yahoo Finance
- Implemented `_fetch_ashare()` method to handle Ashare-specific logic
- Added Ashare fetcher initialization in the constructor
- Updated fetch method to include 'ashare' as a valid source option

### 3. Stock Screener Enhancement (`quant_trade_a_share/screeners/stock_screener.py`)
- Added AshareDataFetcher initialization
- Enhanced `fetch_stock_data()` method to support `data_source='ashare'` parameter
- Added proper data source selection with 'auto' fallback mechanism
- Maintains compatibility with enhanced technical indicators

### 4. CLI Interface Updates (`cli_interface.py`)
- Updated `get_data()` to allow user to select data source including Ashare
- Enhanced `analyze_stock()` to support Ashare data source selection
- Improved `run_strategy()` with data source selection capability
- Updated `gen_signals()` to support Ashare data source
- Enhanced `calc_indicators()` to use data source selection

## Supported Features

### Data Frequencies
- Daily data ('1d')
- Weekly data ('1w')
- Monthly data ('1M')
- Minute data ('1m', '5m', '15m', '30m', '60m')

### Stock Code Formats
- Automatic conversion between formats:
  - `000001.XSHG` → `sh000001`
  - `000001.XSHE` → `sz000001`
  - Raw codes with automatic prefix detection

### Error Handling
- Robust fallback mechanisms when primary data source fails
- Graceful degradation to alternative sources
- Comprehensive error logging and user notifications

## Technical Indicators
All standard technical indicators are calculated when using Ashare data:
- Moving averages (MA5, MA10, MA20, MA30)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume indicators
- Volatility measures
- And many more advanced indicators

## Benefits of Ashare Integration

1. **Multiple Data Sources**: Provides redundancy with both Tencent and Sina APIs
2. **Reliability**: Fallback mechanisms ensure data availability
3. **Performance**: Efficient data retrieval with standardized output
4. **Compatibility**: Seamless integration with existing system components
5. **Quality**: High-quality historical data with proper OHLCV format

## Testing Results
The integration has been thoroughly tested and confirmed to work with major A-share stocks:
- sh600023 (China Life Insurance)
- sz000001 (Ping An Bank)
- sh600519 (Kweichow Moutai)

All tests confirm successful data retrieval, technical indicator calculation, and compatibility with existing system components.

## Usage
Users can now select 'ashare' as a data source in the interactive CLI or use the API directly:
```python
# Via DataFetcher
data = data_fetcher.fetch('sh600023', '2025-01-01', '2025-12-31', source='ashare')

# Via StockScreener
data = screener.fetch_stock_data('sh600023', period='180', data_source='ashare')
```

The Ashare data source provides a robust, reliable addition to the existing data source options in the A-Share market analysis system.