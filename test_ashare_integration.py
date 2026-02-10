#!/usr/bin/env python3
"""
Test script for Ashare data source integration
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.data.ashare_data_fetcher import AshareDataFetcher
from quant_trade_a_share.data.data_fetcher import DataFetcher
from quant_trade_a_share.screeners.stock_screener import StockScreener

def test_ashare_integration():
    """
    Test the integration of Ashare data source with the existing system
    """
    print("=" * 60)
    print("🧪 Testing Ashare Data Source Integration")
    print("=" * 60)

    # Test 1: Test the Ashare fetcher directly
    print("\n1️⃣  Testing Ashare Data Fetcher Directly...")
    ashare_fetcher = AshareDataFetcher()

    # Test with some common A-share symbols
    test_symbols = ['sh600023', 'sz000001', 'sh600519']

    for symbol in test_symbols:
        print(f"\n   Testing {symbol}...")
        data = ashare_fetcher.fetch_stock_data(symbol, days=30)
        if data is not None and not data.empty:
            print(f"   ✅ Retrieved {len(data)} records for {symbol}")
            print(f"   📊 Data columns: {list(data.columns)}")
            print(f"   📈 Date range: {data.index[0]} to {data.index[-1]}")
            print(f"   💰 Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
        else:
            print(f"   ❌ Failed to retrieve data for {symbol}")

    # Test 2: Test integration with DataFetcher
    print("\n2️⃣  Testing Ashare Data Source Integration with DataFetcher...")
    data_fetcher = DataFetcher()

    for symbol in test_symbols:
        print(f"\n   Testing {symbol} via DataFetcher...")
        data = data_fetcher.fetch(symbol,
                                 start_date='2024-01-01',
                                 end_date='2024-02-01',
                                 source='ashare')
        if data is not None and not data.empty:
            print(f"   ✅ Retrieved {len(data)} records for {symbol}")
            print(f"   📊 Data columns: {list(data.columns)}")
        else:
            print(f"   ❌ Failed to retrieve data for {symbol} via DataFetcher")

    # Test 3: Test integration with StockScreener
    print("\n3️⃣  Testing Ashare Data Source Integration with StockScreener...")
    screener = StockScreener()

    for symbol in test_symbols:
        print(f"\n   Testing {symbol} via StockScreener with Ashare...")
        data = screener.fetch_stock_data(symbol, period='30', data_source='ashare')
        if data is not None and not data.empty:
            print(f"   ✅ Retrieved {len(data)} records for {symbol}")
            print(f"   📊 Data columns: {list(data.columns)}")
            if 'rsi' in data.columns and not data['rsi'].empty:
                print(f"   📈 RSI indicator available: {data['rsi'].iloc[-1]:.2f}")
        else:
            print(f"   ❌ Failed to retrieve data for {symbol} via StockScreener")

    # Test 4: Compare different data sources
    print("\n4️⃣  Comparing Different Data Sources for SH600023...")
    symbol = 'sh600023'

    sources_to_test = ['ashare', 'eastmoney']
    source_data = {}

    for source in sources_to_test:
        print(f"\n   Testing {source} source...")
        if source == 'ashare':
            data = screener.fetch_stock_data(symbol, period='10', data_source='ashare')
        else:  # eastmoney
            data = screener.fetch_stock_data(symbol, period='10', data_source='eastmoney')

        source_data[source] = data

        if data is not None and not data.empty:
            print(f"   ✅ {source}: {len(data)} records, price: {data['close'].iloc[-1]:.2f}")
        else:
            print(f"   ❌ {source}: Failed to retrieve data")

    print("\n✅ Ashare Data Source Integration Test Complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_ashare_integration()