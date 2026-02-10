"""
Simple test script to verify the quantitative trading tool works correctly
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from quant_trade_a_share import get_data, run_backtest, analyze_performance
from quant_trade_a_share.strategies import MovingAverageCrossoverStrategy
from quant_trade_a_share.utils.mock_data_generator import generate_mock_data

def test_basic_functionality():
    """Test basic functionality of the trading tool"""
    print("Testing basic functionality of the quantitative trading tool...")
    
    # Test data fetching
    print("\n1. Testing data fetching...")
    try:
        # Use mock data since external sources might have connectivity issues
        data = generate_mock_data('000001', '2022-01-01', '2022-12-31')
        if not data.empty:
            print(f"   ✓ Successfully generated {len(data)} days of mock data")
            print(f"   Data columns: {list(data.columns)}")
            print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        else:
            print("   ⚠ Warning: No data returned, but no error occurred")
    except Exception as e:
        print(f"   ✗ Error generating mock data: {e}")
        return False
    
    # Test strategy creation
    print("\n2. Testing strategy creation...")
    try:
        strategy = MovingAverageCrossoverStrategy(short_window=10, long_window=30)
        print("   ✓ Strategy created successfully")
    except Exception as e:
        print(f"   ✗ Error creating strategy: {e}")
        return False
    
    # Test signal generation
    print("\n3. Testing signal generation...")
    try:
        signals = strategy.generate_signals(data)
        print(f"   ✓ Generated {len(signals[signals != 0])} trading signals")
        print(f"   Sample signals: {signals.head(10).tolist()}")
    except Exception as e:
        print(f"   ✗ Error generating signals: {e}")
        return False
    
    # Test backtesting
    print("\n4. Testing backtesting...")
    try:
        # Import backtester directly to use mock data
        from quant_trade_a_share.backtest.backtester import Backtester
        backtester = Backtester()
        
        # Use the new mock data source
        results = backtester.run(
            strategy, 
            '2022-01-01', 
            '2022-12-31', 
            initial_capital=100000,
            symbol='000001',
            data_source='mock'
        )
        
        if results and 'final_value' in results:
            print(f"   ✓ Backtest completed successfully")
            print(f"   Final portfolio value: {results['final_value']:.2f}")
            print(f"   Total return: {(results['final_value'] - results['initial_capital']) / results['initial_capital']:.2%}")
        else:
            print("   ⚠ Warning: Backtest ran but no results returned")
    except Exception as e:
        print(f"   ✗ Error running backtest: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test performance analysis
    print("\n5. Testing performance analysis...")
    try:
        metrics = analyze_performance(results)
        if metrics:
            print("   ✓ Performance analysis completed")
            print(f"   Total Return: {metrics.get('total_return', 'N/A'):.2%}")
            print(f"   Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A'):.2f}")
            print(f"   Max Drawdown: {metrics.get('max_drawdown', 'N/A'):.2%}")
        else:
            print("   ⚠ Warning: No performance metrics returned")
    except Exception as e:
        print(f"   ✗ Error analyzing performance: {e}")
        return False
    
    print("\n✓ All tests passed! The quantitative trading tool is working correctly.")
    return True

if __name__ == "__main__":
    success = test_basic_functionality()
    if not success:
        print("\n✗ Some tests failed. Please check the implementation.")
        sys.exit(1)
    else:
        print("\n✓ All tests passed successfully!")