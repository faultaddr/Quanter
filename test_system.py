#!/usr/bin/env python3
"""
A-Share Quantitative Trading System - Quick Test
"""

import sys
import os
import pandas as pd
import numpy as np

# Add the project root to the path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("🔍 Testing A-Share Quantitative Trading System Components")
print("="*60)

# Test 1: Check if basic imports work
print("\n1️⃣  Testing basic imports...")
try:
    from quant_trade_a_share.data.data_fetcher import DataFetcher
    print("✅ DataFetcher imported successfully")

    from quant_trade_a_share.strategies.strategy_tools import StrategyManager
    print("✅ StrategyManager imported successfully")

    from quant_trade_a_share.strategies.qlib_strategies import QlibStrategyManager
    print("✅ QlibStrategyManager imported successfully")

    from multi_factor_strategy_template import MultiFactorStrategy
    print("✅ MultiFactorStrategy imported successfully")

except ImportError as e:
    print(f"❌ Import error: {e}")

# Test 2: Check if Qlib is properly configured
print("\n2️⃣  Testing Qlib availability...")
try:
    import qlib
    print(f"✅ Qlib version: {qlib.__version__}")

    # Try initializing Qlib
    from qlib.constant import REG_CN
    print("✅ Qlib constants imported successfully")

except ImportError as e:
    print(f"❌ Qlib import error: {e}")

# Test 3: Test DataFetcher
print("\n3️⃣  Testing DataFetcher...")
try:
    fetcher = DataFetcher()
    print("✅ DataFetcher instantiated successfully")
    print(f"   Available methods: {[method for method in dir(fetcher) if not method.startswith('_')]}")
except Exception as e:
    print(f"❌ DataFetcher instantiation error: {e}")

# Test 4: Test StrategyManager
print("\n4️⃣  Testing StrategyManager...")
try:
    strategy_mgr = StrategyManager()
    print("✅ StrategyManager instantiated successfully")
    print(f"   Available strategies: {strategy_mgr.get_strategy_names()}")
except Exception as e:
    print(f"❌ StrategyManager instantiation error: {e}")

# Test 5: Test QlibStrategyManager
print("\n5️⃣  Testing QlibStrategyManager...")
try:
    qlib_strategy_mgr = QlibStrategyManager()
    print("✅ QlibStrategyManager instantiated successfully")
    print(f"   Available Qlib strategies: {qlib_strategy_mgr.get_strategy_names()}")
except Exception as e:
    print(f"❌ QlibStrategyManager instantiation error: {e}")

# Test 6: Test MultiFactorStrategy
print("\n6️⃣  Testing MultiFactorStrategy...")
try:
    multi_factor_strategy = MultiFactorStrategy()
    print("✅ MultiFactorStrategy instantiated successfully")
    print(f"   Default universe: {multi_factor_strategy.universe}")
except Exception as e:
    print(f"❌ MultiFactorStrategy instantiation error: {e}")

# Test 7: Create mock data and test factor calculation
print("\n7️⃣  Testing factor calculation with mock data...")
try:
    # Create sample data
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    np.random.seed(42)
    prices = 100 + np.cumsum(np.random.randn(100) * 0.5)

    sample_data = pd.DataFrame({
        'open': prices * (1 + np.random.randn(100) * 0.01),
        'high': prices * (1 + abs(np.random.randn(100)) * 0.02),
        'low': prices * (1 - abs(np.random.randn(100)) * 0.02),
        'close': prices,
        'volume': np.random.randint(1000000, 5000000, 100)
    }, index=dates)

    # Calculate factors
    factor_data = multi_factor_strategy.calculate_all_factors(sample_data)
    print(f"✅ Factor calculation successful: {len(factor_data.columns)} factors calculated")
    print(f"   Sample factors: {list(factor_data.columns[:10])}")

except Exception as e:
    print(f"❌ Factor calculation error: {e}")
    import traceback
    traceback.print_exc()

# Test 8: Test signal generation
print("\n8️⃣  Testing signal generation...")
try:
    signals = multi_factor_strategy.generate_signals(factor_data)
    print(f"✅ Signal generation successful: {len(signals)} signal sets generated")
    print(f"   Sample signals shape: {signals.shape}")

except Exception as e:
    print(f"❌ Signal generation error: {e}")

print("\n" + "="*60)
print("✅ System health check completed!")
print("\n💡 Next steps:")
print("   1. Run the full system with: python cli_interface.py --mode interactive")
print("   2. Or run specific analysis tasks")
print("   3. The system supports:")
print("      - Multiple data sources (Ashare, EastMoney, Tushare, BaoStock)")
print("      - 100+ technical indicators")
print("      - Qlib-based advanced strategies")
print("      - Multi-factor analysis")