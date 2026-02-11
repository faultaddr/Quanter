# A-Share Stock Signals Terminal Visualization Tool

## Overview
This tool analyzes and displays stock signals from multiple quantitative strategies in a terminal-based interface. It connects to the existing signals database and generates new signals using various trading strategies.

## Features
- Displays all signals in the database with a formatted table
- Generates new signals using multiple quantitative strategies
- Shows price charts as ASCII art in the terminal
- Provides strategy-specific signal summaries
- Connects to multiple data sources (Ashare, EastMoney)

## How it Works

### 1. Signal Generation Strategies
The tool uses several trading strategies to generate buy/sell signals:
- **alphas101**: Quantitative factor-based strategy using alpha formulas
- **ml_gbdt**: Machine learning gradient boosting decision tree model
- **technical**: Technical indicator-based strategy
- **ensemble**: Combined strategy using multiple models
- **ma_crossover**: Moving average crossover strategy

### 2. Data Flow
1. Fetches stock data from either EastMoney or Ashare APIs
2. Applies various trading strategies to the data
3. Generates buy/sell signals based on each strategy
4. Stores signals in a SQLite database (`signals.db`)
5. Displays signals in terminal with ASCII charts

### 3. Terminal Display
- ASCII charts showing price movement
- Formatted table of all signals with timestamps
- Strategy-specific breakdown of signal types
- Summary statistics of signals

## Sample Output Explanation

From the run, we can see:
- **sh600519 (贵州茅台)**: Various strategies produced signals with different frequencies
- **sz000858 (五粮液)**: Similar pattern with alphas101 generating the most signals
- Signal distribution: Both stocks had more SELL than BUY signals from the alphas101 strategy
- The system correctly prioritized Ashare as primary data source when EastMoney failed

## Database Integration
All generated signals are stored in `signals.db` with the following fields:
- `id`: Signal ID
- `symbol`: Stock symbol
- `signal_type`: BUY/SELL/HOLD
- `strategy`: Strategy that generated the signal
- `timestamp`: When the signal was generated
- `price`: Price at the time of signal
- `reason`: Explanation for the signal

## Technical Details
- Uses Qlib-based strategies for advanced analysis
- Implements 100+ factor analysis from advanced strategies
- Handles API failures gracefully with fallback data sources
- Provides comprehensive error handling
- Maintains signal history in persistent storage

## Usage
The tool can be extended with command-line options to analyze specific stocks or time periods, but currently runs a default analysis on top A-share stocks.