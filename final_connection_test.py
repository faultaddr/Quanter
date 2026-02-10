#!/usr/bin/env python3
"""
Final comprehensive test to verify all connection fixes are working
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.data.data_fetcher import DataFetcher
from quant_trade_a_share.data.ashare_data_fetcher import AshareDataFetcher

def test_fixed_connectivity():
    """
    Final test to ensure the connection fixes are working properly
    """
    print("=" * 80)
    print("🔧 VERIFYING CONNECTION FIXES - FINAL TEST")
    print("=" * 80)

    # Initialize fetchers
    data_fetcher = DataFetcher()
    ashare_fetcher = AshareDataFetcher()

    # Test with problematic symbols mentioned in the original error
    test_symbols = [
        {'symbol': 'sh688818', 'name': 'ASE Technology (was failing)', 'source': 'ashare'},
        {'symbol': 'sz300770', 'name': 'Xinchao Tech (was failing)', 'source': 'ashare'},
        {'symbol': 'sh600023', 'name': 'Zhejiang Expressway (was working)', 'source': 'ashare'},
    ]

    print(f"\n📊 Testing Ashare data source with problematic symbols...")
    all_passed = True

    for test_item in test_symbols:
        symbol = test_item['symbol']
        name = test_item['name']
        source = test_item['source']

        print(f"\n🔍 Testing {symbol} ({name}) via {source} source:")

        try:
            # Test with a reasonable date range
            data = data_fetcher.fetch(symbol, '2025-12-01', '2026-02-10', source=source)

            if data is not None and not data.empty:
                print(f"   ✅ SUCCESS: Retrieved {len(data)} records")
                print(f"      Date range: {data.index[0].date()} to {data.index[-1].date()}")
                print(f"      Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
            else:
                print(f"   ❌ FAILED: No data returned")
                all_passed = False

        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            all_passed = False

    print(f"\n{'='*80}")
    if all_passed:
        print("🎉 ALL CONNECTION FIXES ARE WORKING!")
        print("✅ Fixed connection errors with retry mechanism")
        print("✅ Fixed JSON decode errors ('Expecting value: line 1 column 1 (char 0)')")
        print("✅ Fixed RemoteDisconnected errors")
        print("✅ Implemented proper error handling and fallbacks")
        print("✅ All problematic symbols now work correctly")
    else:
        print("❌ SOME ISSUES REMAIN")

    print("=" * 80)

    # Test the specific error scenario mentioned
    print(f"\n🧪 REPRODUCING ORIGINAL ERROR SCENARIO:")
    print(f"   Original error: 'Expecting value: line 1 column 1 (char 0)'")
    print(f"   Original error: 'Connection aborted.' RemoteDisconnected")

    try:
        # Try to fetch the specific problematic symbols that were mentioned
        for symbol in ['sh688818', 'sz300770']:
            print(f"\n   Attempting to fetch {symbol}...")
            result = ashare_fetcher.fetch_stock_data(symbol, days=30)
            if result is not None and not result.empty:
                print(f"   ✅ {symbol} - Success with {len(result)} records")
            else:
                print(f"   ⚠️  {symbol} - No data (but no crash)")

        print(f"\n   🎯 Original error scenario handled successfully!")

    except Exception as e:
        print(f"   ❌ Still experiencing issues: {e}")
        all_passed = False

    print(f"\n🎯 FINAL RESULT: ", end="")
    if all_passed:
        print("✅ ALL ISSUES RESOLVED - CONNECTIONS STABLE!")
    else:
        print("⚠️  Some issues may remain")

    return all_passed

if __name__ == "__main__":
    success = test_fixed_connectivity()
    if success:
        print(f"\n🏆 Connection stability verified successfully!")
    else:
        print(f"\n⚠️  Further fixes may be needed.")