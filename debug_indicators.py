#!/usr/bin/env python3
"""
Debug script to test technical indicator calculations with sample data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from quant_trade_a_share.utils.mytt_indicators import calculate_mytt_indicators

def generate_sample_data():
    """
    Generate sample OHLCV data for testing
    """
    print("Generating sample stock data...")

    # Create date range
    dates = pd.date_range(start=datetime.now() - timedelta(days=180), end=datetime.now(), freq='D')
    dates = dates[dates.weekday < 5]  # Only weekdays

    # Generate realistic OHLCV data
    n = len(dates)
    base_price = 50.0

    # Generate random walk prices
    returns = np.random.normal(0.001, 0.02, n)  # Mean return 0.1%, std 2%
    prices = [base_price]

    for ret in returns[1:]:
        new_price = prices[-1] * (1 + ret)
        # Ensure price stays reasonable
        new_price = max(new_price, base_price * 0.5)  # Don't drop below 50%
        prices.append(new_price)

    # Add some volatility clustering to make it more realistic
    for i in range(1, len(prices)):
        if i > 1 and abs(prices[i] - prices[i-1])/prices[i-1] > 0.05:  # Big move
            # Adjust next few prices to create mean reversion or momentum
            for j in range(min(3, len(prices)-i)):
                adj = np.random.normal(0, 0.01)
                prices[i+j] = prices[i+j-1] * (1 + adj)

    # Create DataFrame
    df = pd.DataFrame({
        'date': dates[:len(prices)],
        'open': prices,
        'close': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'high': [max(o, c) * (1 + abs(np.random.normal(0, 0.01))) for o, c in zip(prices, [p * (1 + np.random.normal(0, 0.005)) for p in prices])],
        'low': [min(o, c) * (1 - abs(np.random.normal(0, 0.01))) for o, c in zip(prices, [p * (1 + np.random.normal(0, 0.005)) for p in prices])],
        'volume': np.random.randint(1000000, 10000000, len(prices))
    })

    # Ensure high >= open/close and low <= open/close
    for i in range(len(df)):
        df.loc[i, 'high'] = max(df.loc[i, 'high'], df.loc[i, 'open'], df.loc[i, 'close'])
        df.loc[i, 'low'] = min(df.loc[i, 'low'], df.loc[i, 'open'], df.loc[i, 'close'])

    df.set_index('date', inplace=True)
    print(f"Generated {len(df)} days of sample data")
    return df

def test_indicator_calculation():
    """
    Test calculating indicators with sample data
    """
    print("Testing technical indicator calculation...")

    # Generate sample data
    sample_data = generate_sample_data()

    print(f"Sample data shape: {sample_data.shape}")
    print(f"Date range: {sample_data.index[0]} to {sample_data.index[-1]}")
    print("\nFirst few rows:")
    print(sample_data.head())

    # Calculate indicators
    print("\nCalculating MyTT indicators...")
    try:
        enhanced_data = calculate_mytt_indicators(sample_data.copy())

        print(f"Enhanced data shape: {enhanced_data.shape}")
        print(f"Number of new columns added: {len(enhanced_data.columns) - len(sample_data.columns)}")

        # Check some key indicators
        key_indicators = ['rsi6', 'rsi12', 'rsi24', 'macd_dif', 'macd_dea', 'macd_bar',
                         'kdj_k', 'kdj_d', 'kdj_j', 'ma5', 'ma10', 'ma20', 'boll_upper',
                         'boll_mid', 'boll_lower', 'cci', 'atr', 'bias6', 'bias12', 'bias24']

        print("\nChecking key indicators:")
        for indicator in key_indicators:
            if indicator in enhanced_data.columns:
                last_value = enhanced_data[indicator].iloc[-1]
                print(f"  {indicator}: {last_value}")
                # Check if it's 0 (which would indicate the issue)
                if last_value == 0:
                    print(f"    ⚠️  Warning: {indicator} is 0!")
            else:
                print(f"  {indicator}: NOT FOUND")

        # Save the enhanced data for further analysis
        enhanced_data.to_csv('/Users/missy/PROJ/Quanter/quant_trade_a_share/enhanced_sample_data.csv')
        print(f"\nEnhanced data saved to enhanced_sample_data.csv")

        return enhanced_data

    except Exception as e:
        print(f"Error calculating indicators: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("Debugging Technical Indicator Calculation")
    print("="*50)

    result = test_indicator_calculation()

    if result is not None:
        print(f"\n✅ Successfully calculated indicators for {len(result)} days of data")
        print("Last row of data with indicators:")
        print(result.iloc[-1][result.columns[result.dtypes != 'object']].round(4))
    else:
        print("\n❌ Failed to calculate indicators")