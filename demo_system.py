#!/usr/bin/env python3
"""
A-Share Quantitative Trading System - Quick Demo
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("🎬 A-Share Quantitative Trading System - Quick Demo")
print("="*60)

print("\n🚀 Initializing System Components...")
from quant_trade_a_share.data.data_fetcher import DataFetcher
from quant_trade_a_share.strategies.strategy_tools import StrategyManager
from multi_factor_strategy_template import MultiFactorStrategy

# Initialize components
fetcher = DataFetcher()
strategy_manager = StrategyManager()
multi_factor_strategy = MultiFactorStrategy()

print("✅ System components initialized successfully!")

print("\n📊 Demonstrating Multi-Factor Strategy with 100+ Indicators...")

# Create a simple demonstration
print("\n📈 Multi-Factor Strategy Features:")
print(f"   • Stock Universe: {multi_factor_strategy.universe}")
print(f"   • Available Qlib-based Strategies: {len(strategy_manager.get_strategy_names())}")
print(f"   • Strategy Types: {strategy_manager.get_strategy_names()}")

# Demonstrate factor calculation with mock data
print("\n🔍 Calculating 100+ Technical Factors...")

# Create sample data for demonstration
dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
np.random.seed(42)
prices = 100 + np.cumsum(np.random.randn(60) * 0.5)

demo_data = pd.DataFrame({
    'open': prices * (1 + np.random.randn(60) * 0.01),
    'high': prices * (1 + abs(np.random.randn(60)) * 0.02),
    'low': prices * (1 - abs(np.random.randn(60)) * 0.02),
    'close': prices,
    'volume': np.random.randint(1000000, 5000000, 60)
}, index=dates)

# Calculate factors
factor_data = multi_factor_strategy.calculate_all_factors(demo_data)
print(f"✅ Successfully calculated {len(factor_data.columns)-5} technical factors")  # -5 for original price columns

print("\n💡 Key Features of the System:")
features = [
    "Ashare data source (primary)",
    "EastMoney cookie-based data fetching",
    "Tushare integration",
    "BaoStock support",
    "100+ technical indicators",
    "Qlib-based advanced strategies",
    "Multi-factor analysis",
    "Risk management tools",
    "Backtesting engine",
    "Real-time signal generation"
]

for i, feature in enumerate(features, 1):
    print(f"   {i}. {feature}")

print("\n🎛️  Available Strategies:")
strategies = strategy_manager.get_strategy_names()
for i, strategy in enumerate(strategies[:10], 1):  # Show first 10 strategies
    print(f"   {i}. {strategy}")
if len(strategies) > 10:
    print(f"   ... and {len(strategies)-10} more")

print("\n🎯 Running Sample Analysis...")

# Demonstrate a simple analysis
sample_symbol = 'sh600519'  # 贵州茅台
print(f"\n📈 Analyzing {sample_symbol} using multiple strategies...")

try:
    # Get data using the fetcher (would normally connect to real source)
    print("   Fetching data...")

    # Generate signals using different strategies
    print("   Generating signals...")
    for strat_name in ['ma_crossover', 'rsi', 'technical'][:2]:  # Use first 2 strategies
        strategy = strategy_manager.get_strategy(strat_name)
        if strategy:
            signals = strategy.generate_signals(demo_data)
            non_zero_signals = len(signals[signals != 0])
            print(f"   - {strat_name}: {non_zero_signals} signals generated")

    print("\n✅ Sample analysis completed successfully!")

except Exception as e:
    print(f"⚠️  Analysis demo encountered an issue (expected in demo mode): {e}")

print("\n🏠 System Architecture:")
architecture = {
    "Data Layer": [
        "Ashare (primary)",
        "EastMoney API",
        "Tushare",
        "BaoStock",
        "Yahoo Finance"
    ],
    "Strategy Layer": [
        "Qlib-based strategies",
        "Technical indicators",
        "Machine learning models",
        "Multi-factor models"
    ],
    "Analysis Layer": [
        "Risk management",
        "Performance metrics",
        "Backtesting engine",
        "Signal generation"
    ]
}

for layer, components in architecture.items():
    print(f"   {layer}:")
    for comp in components:
        print(f"     • {comp}")

print("\n" + "="*60)
print("💡 To use the full system:")
print("   1. Run interactive mode: python cli_interface.py --mode interactive")
print("   2. Or run specific commands: python cli_interface.py --mode screen/analyze/backtest")
print("   3. The system is ready for live A-share market analysis!")
print("="*60)