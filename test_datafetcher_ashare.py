#!/usr/bin/env python3
"""
Test script specifically for DataFetcher Ashare integration
"""
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quant_trade_a_share.data.data_fetcher import DataFetcher

def test_datafetcher_ashare():
    """
    Test DataFetcher specifically with Ashare source using appropriate date range
    """
    print("=" * 60)
    print("🧪 Testing DataFetcher with Ashare Source")
    print("=" * 60)

    # Initialize DataFetcher
    data_fetcher = DataFetcher()

    # Use recent dates that would align with available data
    # Since Ashare data is from late 2025 to early 2026, let's use those dates
    recent_symbol = 'sh600023'
    start_date = '2025-12-29'  # Start of available data
    end_date = '2026-02-10'    # End of available data

    print(f"\nTesting {recent_symbol} with date range {start_date} to {end_date}...")

    # Test fetching data with Ashare source
    data = data_fetcher.fetch(recent_symbol, start_date, end_date, source="ashare")

    if data is not None and not data.empty:
        print(f"✅ Successfully retrieved {len(data)} records for {recent_symbol} via DataFetcher Ashare source")
        print(f"📊 Data columns: {list(data.columns)}")
        print(f"📈 Date range: {data.index[0]} to {data.index[-1]}")
        print(f"💰 Price range: {data['close'].min():.2f} - {data['close'].max():.2f}")
        print(f"📊 Sample data:\n{data.head()}")
    else:
        print(f"❌ Failed to retrieve data for {recent_symbol} via DataFetcher Ashare source")
        if data is None:
            print("   Returned data is None")
        else:
            print(f"   Returned dataframe is empty with shape: {data.shape}")

    # Also test with broader date range that includes the available data
    broader_start = '2025-12-01'  # Before available data starts
    broader_end = '2026-02-28'    # After available data ends

    print(f"\nTesting {recent_symbol} with broader date range {broader_start} to {broader_end}...")

    # Test fetching data with Ashare source
    data2 = data_fetcher.fetch(recent_symbol, broader_start, broader_end, source="ashare")

    if data2 is not None and not data2.empty:
        print(f"✅ Successfully retrieved {len(data2)} records for {recent_symbol} via DataFetcher Ashare source (broad range)")
        print(f"📊 Date range in result: {data2.index[0]} to {data2.index[-1]}")
        print(f"💰 Price range: {data2['close'].min():.2f} - {data2['close'].max():.2f}")
    else:
        print(f"❌ Failed to retrieve data for {recent_symbol} via DataFetcher Ashare source (broad range)")

    print("\n✅ DataFetcher Ashare Integration Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    test_datafetcher_ashare()