#!/usr/bin/env python3
"""
Analysis script for stock sh601777 based on the past 360 days of data.
"""

import sys
import os
import pandas as pd
from datetime import datetime, timedelta

# Add project root to Python path
sys.path.insert(0, '/root/CuferPan/quanttool')

from quanttool.infrastructure.data_providers.data_fetcher import create_data_fetcher_with_credentials


def analyze_stock():
    """Analyze stock sh601777 based on the past 360 days of data."""

    print("Starting analysis for stock: sh601777 (国投资本)")
    print(f"Current date: {datetime.now().strftime('%Y-%m-%d')}")

    # Create the data fetcher with credentials
    print("\nInitializing Enhanced DataFetcher...")
    fetcher = create_data_fetcher_with_credentials()
    fetcher.initialize()

    print("DataFetcher initialized successfully!")

    # Define date range: past 360 days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=360)

    print(f"Fetching data for period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")

    # Stock symbol - converting to appropriate format for Tushare
    symbols = ["601777.SH"]  # Tushare format for Shanghai stocks

    print(f"\nFetching historical data for {symbols[0]}...")
    data = fetcher.get_bars(symbols, start_date, end_date)

    if symbols[0] in data and not data[symbols[0]].empty:
        df = data[symbols[0]]
        print(f"Retrieved {len(df)} records from {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Display sample data
        print("\nSample data (first 5 records):")
        print(df.head())

        print("\nSample data (last 5 records):")
        print(df.tail())

        # Basic statistics
        print("\nBasic Statistics:")
        print(f"Total records: {len(df)}")
        print(f"Start date: {df['timestamp'].min()}")
        print(f"End date: {df['timestamp'].max()}")
        print(f"Opening price range: {df['open'].min():.2f} - {df['open'].max():.2f}")
        print(f"Closing price range: {df['close'].min():.2f} - {df['close'].max():.2f}")
        print(f"Highest price: {df['high'].max():.2f}")
        print(f"Lowest price: {df['low'].min():.2f}")
        print(f"Average daily volume: {df['volume'].mean():,.0f}")
        print(f"Total traded volume: {df['volume'].sum():,.0f}")

        # Calculate additional metrics
        df_sorted = df.sort_values('timestamp')
        df_sorted['daily_return'] = df_sorted['close'].pct_change()
        df_sorted['price_change'] = df_sorted['close'] - df_sorted['open']
        df_sorted['price_range'] = df_sorted['high'] - df_sorted['low']

        print(f"\nReturn Analysis:")
        print(f"First closing price: {df_sorted['close'].iloc[0]:.2f}")
        print(f"Last closing price: {df_sorted['close'].iloc[-1]:.2f}")
        total_return = (df_sorted['close'].iloc[-1] / df_sorted['close'].iloc[0] - 1) * 100
        print(f"Total return over the period: {total_return:.2f}%")
        print(f"Average daily return: {df_sorted['daily_return'].mean()*100:.3f}%")
        print(f"Daily return volatility (std): {df_sorted['daily_return'].std()*100:.3f}%")

        # Find highest and lowest prices and when they occurred
        max_close_idx = df_sorted['close'].idxmax()
        min_close_idx = df_sorted['close'].idxmin()

        print(f"\nPrice Extremes:")
        print(f"Highest closing price: {df_sorted.loc[max_close_idx, 'close']:.2f} on {df_sorted.loc[max_close_idx, 'timestamp'].date()}")
        print(f"Lowest closing price: {df_sorted.loc[min_close_idx, 'close']:.2f} on {df_sorted.loc[min_close_idx, 'timestamp'].date()}")

        # Recent trend (last 30 days)
        recent_data = df_sorted.tail(30)
        if len(recent_data) >= 2:
            recent_return = (recent_data['close'].iloc[-1] / recent_data['close'].iloc[0] - 1) * 100
            print(f"\nRecent Trend (last 30 days): {recent_return:.2f}%")

        # Volume analysis
        print(f"\nVolume Analysis:")
        print(f"Average daily volume: {df_sorted['volume'].mean():,.0f}")
        print(f"Median daily volume: {df_sorted['volume'].median():,.0f}")
        max_vol_idx = df_sorted['volume'].idxmax()
        print(f"Highest volume: {df_sorted.loc[max_vol_idx, 'volume']:,.0f} on {df_sorted.loc[max_vol_idx, 'timestamp'].date()}")

        # Moving averages
        df_sorted['ma_20'] = df_sorted['close'].rolling(window=20).mean()
        df_sorted['ma_50'] = df_sorted['close'].rolling(window=50).mean()
        df_sorted['ma_200'] = df_sorted['close'].rolling(window=200).mean()

        print(f"\nMoving Averages (last values):")
        print(f"MA-20: {df_sorted['ma_20'].iloc[-1]:.2f}")
        print(f"MA-50: {df_sorted['ma_50'].iloc[-1]:.2f}")
        print(f"MA-200: {df_sorted['ma_200'].iloc[-1]:.2f}")

        # Show recent data with moving averages
        ma_cols = ['ma_20', 'ma_50', 'ma_200']
        cols_to_show = ['timestamp', 'close', 'volume'] + [col for col in ma_cols if col in recent_data.columns]
        print(f"\nRecent data with moving averages:")
        print(recent_data[cols_to_show].round(2))

        # Export to CSV for further analysis if needed
        output_file = f"sh601777_analysis_{datetime.now().strftime('%Y%m%d')}.csv"
        df_sorted.to_csv(output_file, index=False)
        print(f"\nFull dataset exported to: {output_file}")

        return df_sorted

    else:
        print(f"No data retrieved for {symbols[0]}")
        print("This could be due to:")
        print("1. Insufficient API permissions")
        print("2. Network connectivity issues")
        print("3. Invalid stock symbol")
        print("4. Data availability limitations")

        # Let's try alternative symbol formats
        alt_symbols = ["601777.SH", "601777", "sh601777"]
        print(f"\nTrying alternative symbol formats: {alt_symbols}")

        for alt_symbol in alt_symbols:
            print(f"Trying symbol: {alt_symbol}")
            alt_data = fetcher.get_bars([alt_symbol], start_date, end_date)

            if alt_symbol in alt_data and not alt_data[alt_symbol].empty:
                df_alt = alt_data[alt_symbol]
                print(f"Success! Retrieved {len(df_alt)} records for {alt_symbol}")
                return df_alt
            else:
                print(f"No data found for {alt_symbol}")

        return None


if __name__ == "__main__":
    analyze_stock()